from __future__ import annotations

import asyncio
import collections
import math
import queue
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple

import pandas as pd

import bleak
import cv2
import customtkinter as ctk
import numpy as np
import tkinter as tk
from cv2_enumerate_cameras import enumerate_cameras
import PIL.Image, PIL.ImageDraw, PIL.ImageFont, PIL.ImageTk
import onnxruntime as ort

from writer_id_onnx import ONNXWriterRegistry
from realtime_writer_id import Segmenter

# ---------------------------------------------------------------------------
# Paths (all relative to this file's directory)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"
_ML_DIR = _HERE / "../../ml"
_MEDIA_DIR = _HERE / "media"

STYLE_ENCODER_PATH = _ONNX_DIR / "style_encoder.onnx"
CHANNEL_MEAN_PATH  = _ML_DIR / "style_channel_mean.npy"
CHANNEL_STD_PATH   = _ML_DIR / "style_channel_std.npy"
DET_PATH           = _ONNX_DIR / "det.onnx"
REC_PATH           = _ONNX_DIR / "languages" / "rec.onnx"
DICT_PATH          = _ONNX_DIR / "languages" / "dict.txt"

# ---------------------------------------------------------------------------
# BLE constants (Seeed Studio MG24 IMU service)
# ---------------------------------------------------------------------------
IMU_SERVICE_UUID = "19B10000-E8F2-537E-4F6C-D104768A1214"
IMU_CHAR_UUID    = "19B10001-E8F2-537E-4F6C-D104768A1214"
PACKET_FMT       = "<BBHIhhhhhh"
PACKET_SIZE      = struct.calcsize(PACKET_FMT)  # 20 bytes
SAMPLE_RATE      = 104

# ---------------------------------------------------------------------------
# Writer palette
# ---------------------------------------------------------------------------
WRITER_PALETTE: List[Tuple[int, int, int]] = [
    (220,  50,  50),
    ( 50, 100, 220),
    ( 50, 180,  50),
    (200, 120,  40),
    (150,  50, 200),
    ( 40, 180, 190),
    (200, 180,  40),
    (200,  50, 150),
]
UNKNOWN_COLOR = (130, 130, 130)

# ---------------------------------------------------------------------------
# BulletinBoard
# ---------------------------------------------------------------------------
@dataclass
class BulletinEvent:
    kind: str          # "pen_segment" | "cv_update" | "writer_enrolled"
    timestamp: float = field(default_factory=time.time)
    writer_id: Optional[str] = None
    sim_score: Optional[float] = None
    cv_type: Optional[str] = None   # "ocr" | "trace"
    text_content: Optional[str] = None


class BulletinBoard:
    MAX_EVENTS = 100

    def __init__(self) -> None:
        self._events: List[BulletinEvent] = []

    def post(self, event: BulletinEvent) -> None:
        self._events.append(event)
        if len(self._events) > self.MAX_EVENTS:
            self._events.pop(0)

    def last_active_writer(
        self, before_ts: Optional[float] = None, window_secs: float = 10.0
    ) -> Optional[str]:
        ts = before_ts if before_ts is not None else time.time()
        for ev in reversed(self._events):
            if ev.kind == "pen_segment" and ev.writer_id and ev.writer_id != "unknown":
                if ts - ev.timestamp <= window_secs:
                    return ev.writer_id
        return None

    def get_events(self) -> List[BulletinEvent]:
        return list(self._events)


# ---------------------------------------------------------------------------
# WriterIdentityEngine
# ---------------------------------------------------------------------------
class WriterIdentityEngine:
    def __init__(
        self,
        onnx_path: Path = STYLE_ENCODER_PATH,
        channel_mean_path: Path = CHANNEL_MEAN_PATH,
        channel_std_path: Path = CHANNEL_STD_PATH,
        unknown_threshold: float = 0.65,
        onnx_provider: str = "CPUExecutionProvider",
    ) -> None:
        self._onnx_path = onnx_path
        self._mean_path = channel_mean_path
        self._std_path  = channel_std_path
        self._threshold = unknown_threshold
        self._provider  = onnx_provider
        self.registry: Optional[ONNXWriterRegistry] = None
        self._loaded = False

    def load(self) -> None:
        channel_mean = np.load(self._mean_path)
        channel_std  = np.load(self._std_path)
        providers = [self._provider]
        try:
            self.registry = ONNXWriterRegistry(
                onnx_path=self._onnx_path,
                channel_mean=channel_mean,
                channel_std=channel_std,
                target_len=96,
                unknown_threshold=self._threshold,
                providers=providers,
            )
        except Exception:
            # CUDA/cuDNN unavailable or misconfigured — fall back silently to CPU
            self._provider = "CPUExecutionProvider"
            self.registry = ONNXWriterRegistry(
                onnx_path=self._onnx_path,
                channel_mean=channel_mean,
                channel_std=channel_std,
                target_len=96,
                unknown_threshold=self._threshold,
                providers=["CPUExecutionProvider"],
            )
        self._loaded = True

    def set_provider(self, provider: str) -> None:
        self._provider = provider
        if self._loaded:
            self._loaded = False
            self.load()

    def predict_segment(self, segment: np.ndarray) -> Tuple[Optional[str], float]:
        if not self._loaded:
            self.load()
        return self.registry.predict_or_unknown(segment)

    def enroll_segment(self, writer_id: str, segment: np.ndarray, momentum: float = 0.95) -> None:
        if not self._loaded:
            self.load()
        self.registry.update_writer(writer_id=writer_id, segment=segment, momentum=momentum)

    def best_match_segment(self, segment: np.ndarray) -> Tuple[Optional[str], float]:
        """Return the best-matching writer and score regardless of the unknown threshold."""
        if not self._loaded:
            self.load()
        if not self.registry or not self.registry.prototypes:
            return None, 0.0
        emb = self.registry._embed_one(segment)
        best_writer, best_sim = None, -1.0
        for wid, proto in self.registry.prototypes.items():
            sim = float(np.dot(emb, proto))
            if sim > best_sim:
                best_sim, best_writer = sim, wid
        return best_writer, float(best_sim)


# ---------------------------------------------------------------------------
# MyVideoCapture
# ---------------------------------------------------------------------------
class MyVideoCapture:
    def __init__(self, video_source: int = 0) -> None:
        # DirectShow avoids the MSMF/CUDA conflict on Windows
        self.vid = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.width  = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.vid.isOpened():
            return False, None
        ok, frame = self.vid.read()
        if ok:
            return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return False, None

    def __del__(self) -> None:
        if self.vid.isOpened():
            self.vid.release()


# ---------------------------------------------------------------------------
# FileVideoCapture  (drop-in for MyVideoCapture, reads from a file at real-time pace)
# ---------------------------------------------------------------------------
class FileVideoCapture:
    def __init__(self, path: str) -> None:
        self.vid = cv2.VideoCapture(str(path))
        if not self.vid.isOpened():
            raise ValueError("Cannot open video file", path)
        self.width  = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.vid.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_interval = 1.0 / fps
        self._next_frame_time = time.monotonic()
        self._exhausted = False

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._exhausted:
            time.sleep(self._frame_interval)  # prevent tight spin after EOF
            return False, None
        now = time.monotonic()
        wait = self._next_frame_time - now
        if wait > 0:
            time.sleep(wait)
        ok, frame = self.vid.read()
        self._next_frame_time = time.monotonic() + self._frame_interval
        if ok:
            return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._exhausted = True
        return False, None

    def __del__(self) -> None:
        if self.vid.isOpened():
            self.vid.release()


# ---------------------------------------------------------------------------
# CSVIMUPlayer  (drop-in for BLE notifications, replays a CSV into _imu_queue)
# ---------------------------------------------------------------------------
class CSVIMUPlayer:
    TARGET_HZ = 75.0
    _CHANNELS = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]

    def __init__(self, csv_path: str, imu_queue: "queue.Queue[np.ndarray]") -> None:
        self._path      = Path(csv_path)
        self._queue     = imu_queue
        self._running   = False
        self._thread: Optional[threading.Thread] = None
        # Extract writer label from filename: recording1_X_... → "X"
        parts = self._path.stem.split("_")
        self._label: Optional[str] = parts[1] if len(parts) > 1 else None

    @property
    def writer_label(self) -> Optional[str]:
        return self._label

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="imu-player")
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _load_and_resample(self) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(self._path, skiprows=2)
        df.columns = df.columns.str.strip()
        t_src = df["t_rel_host_s"].values.astype(np.float64)
        t_new = np.arange(0.0, t_src[-1], 1.0 / self.TARGET_HZ)
        resampled = np.empty((len(t_new), 6), dtype=np.float32)
        for i, ch in enumerate(self._CHANNELS):
            resampled[:, i] = np.interp(t_new, t_src, df[ch].values).astype(np.float32)
        return t_new, resampled

    def _run(self) -> None:
        t_grid, samples = self._load_and_resample()
        t0 = time.monotonic()
        for t_target, sample in zip(t_grid, samples):
            if not self._running:
                break
            deadline = t0 + float(t_target)
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            try:
                self._queue.put_nowait(sample)
            except queue.Full:
                pass


# ---------------------------------------------------------------------------
# BLEManager
# ---------------------------------------------------------------------------
class BLEManager:
    def __init__(self, imu_queue: "queue.Queue[np.ndarray]") -> None:
        self._imu_queue = imu_queue
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ble-loop")
        self._thread.start()
        self._client: Optional[bleak.BleakClient] = None
        self._connected = threading.Event()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    # --- public (called from main thread, return concurrent.futures.Future) ---

    def scan(self, timeout: float = 5.0):
        return asyncio.run_coroutine_threadsafe(self._scan_async(timeout), self._loop)

    def connect(self, device: bleak.BLEDevice, on_disconnect: Optional[Callable] = None):
        return asyncio.run_coroutine_threadsafe(
            self._connect_async(device, on_disconnect=on_disconnect), self._loop
        )

    def disconnect(self):
        return asyncio.run_coroutine_threadsafe(self._disconnect_async(), self._loop)

    def is_connected(self) -> bool:
        return self._connected.is_set()

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)

    # --- async internals ---

    async def _scan_async(self, timeout: float) -> List[bleak.BLEDevice]:
        # Filter by service UUID so we only surface the pen even if it hasn't
        # broadcast a human-readable name yet.
        seen: dict[str, bleak.BLEDevice] = {}

        def _cb(device: bleak.BLEDevice, _adv) -> None:
            seen[device.address] = device

        async with bleak.BleakScanner(
            detection_callback=_cb,
            service_uuids=[IMU_SERVICE_UUID],
        ):
            await asyncio.sleep(timeout)

        # Fall back to a name-based sweep if the UUID filter found nothing
        # (some Windows BLE stacks don't pass service UUIDs in advertisements).
        if not seen:
            all_devices = await bleak.BleakScanner.discover(timeout=timeout)
            for d in all_devices:
                seen[d.address] = d

        return list(seen.values())

    async def _connect_async(
        self,
        device: bleak.BLEDevice,
        on_disconnect: Optional[Callable] = None,
        retries: int = 3,
    ) -> bool:
        for attempt in range(retries):
            try:
                self._client = bleak.BleakClient(
                    device,
                    disconnected_callback=lambda _: self._on_disconnect(on_disconnect),
                )
                await self._client.connect(timeout=10.0)

                # stop_notify first in case the device still thinks it's subscribed
                try:
                    await self._client.stop_notify(IMU_CHAR_UUID)
                except Exception:
                    pass
                await self._client.start_notify(IMU_CHAR_UUID, self._notification_handler)

                self._connected.set()
                return True
            except Exception:
                self._connected.clear()
                if attempt < retries - 1:
                    await asyncio.sleep(1.5)
        return False

    def _on_disconnect(self, on_disconnect_cb: Optional[Callable]) -> None:
        self._connected.clear()
        if on_disconnect_cb:
            on_disconnect_cb()

    async def _disconnect_async(self) -> None:
        if self._client and self._client.is_connected:
            try:
                await self._client.stop_notify(IMU_CHAR_UUID)
                await self._client.disconnect()
            except Exception:
                pass
        self._connected.clear()

    async def _notification_handler(self, sender, data: bytearray) -> None:
        if len(data) != PACKET_SIZE:
            return
        if data[0] != 0xAA or data[1] != 0x55:
            return
        try:
            vals = struct.unpack(PACKET_FMT, bytes(data))
        except struct.error:
            return
        _, _, _, _, gx10, gy10, gz10, ax_mg, ay_mg, az_mg = vals
        sample6 = np.array(
            [ax_mg / 1000.0, ay_mg / 1000.0, az_mg / 1000.0,
             gx10  /   10.0, gy10  /   10.0, gz10  /   10.0],
            dtype=np.float32,
        )
        try:
            self._imu_queue.put_nowait(sample6)
        except queue.Full:
            pass


# ---------------------------------------------------------------------------
# ROISelector
# ---------------------------------------------------------------------------
class ROISelector:
    COLORS = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]

    def __init__(
        self,
        canvas: tk.Canvas,
        on_complete: Callable[[np.ndarray], None],
    ) -> None:
        self._canvas = canvas
        self._on_complete = on_complete
        self._corners: List[Tuple[int, int]] = []
        self._active = False
        self._frame_w = 1
        self._frame_h = 1
        self._canvas_w = 1
        self._canvas_h = 1

    def start(self) -> None:
        self._corners = []
        self._active = True
        self._canvas.bind("<Button-1>", self._on_click)
        self._redraw()

    def cancel(self) -> None:
        self._active = False
        self._canvas.unbind("<Button-1>")
        self._canvas.delete("roi_overlay")

    def update_frame_size(self, fw: int, fh: int, cw: int, ch: int) -> None:
        self._frame_w = fw
        self._frame_h = fh
        self._canvas_w = cw
        self._canvas_h = ch

    def _on_click(self, event: tk.Event) -> None:
        if not self._active:
            return
        self._corners.append((event.x, event.y))
        self._redraw()
        if len(self._corners) == 4:
            self._active = False
            self._canvas.unbind("<Button-1>")
            sx = self._frame_w / max(self._canvas_w, 1)
            sy = self._frame_h / max(self._canvas_h, 1)
            pts = np.array(
                [[cx * sx, cy * sy] for cx, cy in self._corners],
                dtype=np.float32,
            )
            self._on_complete(pts)

    def _redraw(self) -> None:
        self._canvas.delete("roi_overlay")
        r = 6
        for i, (cx, cy) in enumerate(self._corners):
            color = self.COLORS[i % len(self.COLORS)]
            self._canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                fill=color, outline="white", width=2, tags="roi_overlay",
            )
        for i in range(1, len(self._corners)):
            x0, y0 = self._corners[i - 1]
            x1, y1 = self._corners[i]
            self._canvas.create_line(
                x0, y0, x1, y1, fill="yellow", width=2, tags="roi_overlay",
            )
        if len(self._corners) == 4:
            x0, y0 = self._corners[3]
            x1, y1 = self._corners[0]
            self._canvas.create_line(
                x0, y0, x1, y1, fill="yellow", width=2, tags="roi_overlay",
            )


# ---------------------------------------------------------------------------
# CVWorker
# ---------------------------------------------------------------------------
class CVWorker:
    PROCESS_INTERVAL      = 0.2   # CV analysis at 5fps
    WARP_WIDTH            = 1280
    STABILITY_FRAMES      = 4     # ticks to set initial reference
    STABILITY_THRESH      = 8.0   # coarse motion gate
    OCR_STABILITY_FRAMES  = 1    # ~0.2 s of stillness required to fire OCR
    OCR_STABILITY_THRESH  = 8.0   # resets on writing motion; passes when writer is still
    CHANGE_COVERAGE_MIN   = 0.001
    DIFF_THRESH           = 30
    OCR_MIN_INTERVAL      = 5.0   # minimum seconds between OCR triggers
    PERSON_BLOB_FRAC      = 0.05  # contour covering >5% of ROI with solidity>0.5 = arm/person

    def __init__(
        self,
        vid: MyVideoCapture,
        display_queue: "queue.Queue[np.ndarray]",
        result_queue: "queue.Queue[dict]",
    ) -> None:
        self._vid      = vid
        self._disp_q   = display_queue
        self._q        = result_queue
        self._lock     = threading.Lock()
        self._corners: Optional[np.ndarray] = None
        self._M:   Optional[np.ndarray] = None
        self._warp_w = self.WARP_WIDTH
        self._warp_h = self.WARP_WIDTH
        self._reference:    Optional[np.ndarray] = None
        self._ref_norm:     Optional[np.ndarray] = None
        self._prev_norm:    Optional[np.ndarray] = None
        self._latest_warped: Optional[np.ndarray] = None
        self._latest_norm:   Optional[np.ndarray] = None
        self._stable_count     = 0
        self._ocr_stable_count = 0
        self._last_change_post = 0.0
        self._last_process     = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def set_roi(self, corners: np.ndarray) -> None:
        with self._lock:
            self._corners = corners
            self._M, self._warp_w, self._warp_h = self._compute_warp(corners)
            self._reference        = None
            self._ref_norm         = None
            self._prev_norm        = None
            self._latest_warped    = None
            self._latest_norm      = None
            self._stable_count     = 0
            self._ocr_stable_count = 0
            self._last_change_post = 0.0

    def clear_roi(self) -> None:
        with self._lock:
            self._corners = None
            self._M = None
            self._reference = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="cv-worker")
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _compute_warp(
        self, corners: np.ndarray
    ) -> Tuple[np.ndarray, int, int]:
        tl, tr, br, bl = corners
        top_w    = np.linalg.norm(tr - tl)
        bot_w    = np.linalg.norm(br - bl)
        left_h   = np.linalg.norm(bl - tl)
        right_h  = np.linalg.norm(br - tr)
        avg_w    = float((top_w + bot_w) / 2.0)
        avg_h    = float((left_h + right_h) / 2.0)
        out_w    = self.WARP_WIDTH
        raw_h    = max(32, out_w * avg_h / max(avg_w, 1.0))
        out_h    = int(math.ceil(raw_h / 32) * 32)
        dst = np.array(
            [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        return M, out_w, out_h

    def _run(self) -> None:
        while self._running:
            ok, frame = self._vid.get_frame()  # blocks at camera's native rate
            if not ok or frame is None:
                continue

            # Always forward the latest frame for display; drop stale ones
            try:
                self._disp_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._disp_q.put_nowait(frame)
            except queue.Full:
                pass

            # CV analysis at 5fps
            now = time.monotonic()
            if now - self._last_process >= self.PROCESS_INTERVAL:
                self._last_process = now
                self._process_cv(frame)

    def _process_cv(self, frame: np.ndarray) -> None:
        with self._lock:
            M      = self._M
            warp_w = self._warp_w
            warp_h = self._warp_h

        if M is None:
            return

        warped   = cv2.warpPerspective(frame, M, (warp_w, warp_h))
        norm_new = self._normalize_illumination(warped)

        inter = 0.0
        if self._prev_norm is not None:
            inter = float(cv2.absdiff(norm_new, self._prev_norm).mean())
            self._stable_count     = self._stable_count     + 1 if inter < self.STABILITY_THRESH     else 0
            self._ocr_stable_count = self._ocr_stable_count + 1 if inter < self.OCR_STABILITY_THRESH else 0
        self._prev_norm = norm_new

        with self._lock:
            self._latest_warped = warped
            self._latest_norm   = norm_new

        # Phase 1 — set reference once the scene is stable.
        if self._reference is None:
            if self._stable_count >= self.STABILITY_FRAMES:
                with self._lock:
                    self._reference = warped.copy()
                    self._ref_norm  = norm_new.copy()
            return

        # Phase 2 — fire OCR when scene has been still for OCR_STABILITY_FRAMES ticks.
        if self._ocr_stable_count < self.OCR_STABILITY_FRAMES:
            return

        with self._lock:
            ref_norm = self._ref_norm.copy()

        diff     = cv2.absdiff(norm_new, ref_norm)
        mask     = (diff > self.DIFF_THRESH).astype(np.uint8)
        coverage = float(mask.mean())

        if coverage < self.CHANGE_COVERAGE_MIN:
            return

        now = time.monotonic()
        if now - self._last_change_post < self.OCR_MIN_INTERVAL:
            return

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if self._has_person_blob(contours, mask):
        #     # Arm/person in the ROI — OCR would read their hand or clothing.
        #     # Don't advance the reference so the diff remains fresh once they leave.
        #     return

        self._last_change_post = now
        with self._lock:
            self._reference = warped.copy()
            self._ref_norm  = norm_new.copy()

        try:
            self._q.put_nowait({"warped": warped, "contours": contours})
        except queue.Full:
            pass

    def get_change_info(self) -> Optional[dict]:
        """Called from main thread when pen goes idle. Returns diff vs reference."""
        with self._lock:
            if self._reference is None or self._latest_warped is None:
                return None
            warped   = self._latest_warped.copy()
            ref_norm = self._ref_norm.copy()

        norm_new = self._normalize_illumination(warped)
        diff     = cv2.absdiff(norm_new, ref_norm)
        mask     = (diff > self.DIFF_THRESH).astype(np.uint8)
        coverage = float(mask.mean())

        if coverage < self.CHANGE_COVERAGE_MIN:
            return None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return {"warped": warped, "contours": contours}

    def advance_reference(self) -> None:
        """Commit latest warped frame as the new reference after OCR completes."""
        with self._lock:
            if self._latest_warped is not None:
                self._reference = self._latest_warped.copy()
                self._ref_norm  = self._latest_norm.copy()

    def _has_person_blob(self, contours: tuple, mask: np.ndarray) -> bool:
        total = mask.shape[0] * mask.shape[1]
        threshold = total * self.PERSON_BLOB_FRAC
        for c in contours:
            area = cv2.contourArea(c)
            if area < threshold:
                continue
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity  = area / (hull_area + 1e-6)
            if solidity > 0.5:
                return True
        return False

    @staticmethod
    def _normalize_illumination(img_rgb: np.ndarray) -> np.ndarray:
        gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        k     = min(201, max(51, (img_rgb.shape[1] // 8) | 1))  # odd, ~1/8 width
        illum = cv2.GaussianBlur(gray, (k, k), 0)
        norm  = gray / (illum + 1.0) * 128.0
        return np.clip(norm, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# PaddleOCRPipeline
# ---------------------------------------------------------------------------
class PaddleOCRPipeline:
    DET_W  = 960
    REC_H  = 48
    DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    DET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        det_path: Path = DET_PATH,
        rec_path: Path = REC_PATH,
        dict_path: Path = DICT_PATH,
        provider: str = "CPUExecutionProvider",
    ) -> None:
        self._det_path  = det_path
        self._rec_path  = rec_path
        self._dict_path = dict_path
        self._provider  = provider
        self._det_sess: Optional[ort.InferenceSession] = None
        self._rec_sess: Optional[ort.InferenceSession] = None
        self._vocab: List[str] = []

    def _load(self) -> None:
        providers = [self._provider]
        self._det_sess = ort.InferenceSession(str(self._det_path), providers=providers)
        self._rec_sess = ort.InferenceSession(str(self._rec_path), providers=providers)
        chars = [line.rstrip("\n") for line in self._dict_path.read_text(encoding="utf-8").splitlines()]
        self._vocab = [""] + chars  # index 0 = blank

    def set_provider(self, provider: str) -> None:
        self._provider = provider
        self._det_sess = None
        self._rec_sess = None

    def _ensure_loaded(self) -> None:
        if self._det_sess is None:
            self._load()

    @staticmethod
    def _enhance(img_rgb: np.ndarray) -> np.ndarray:
        """CLAHE on the L channel to even out whiteboard illumination."""
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    def run(self, img_rgb: np.ndarray) -> List[Dict]:
        self._ensure_loaded()
        img_rgb = self._enhance(img_rgb)
        boxes = self._detect(img_rgb)
        if not boxes:
            return []
        texts = self._recognize(img_rgb, boxes)
        results = []
        for box, (text, conf) in zip(boxes, texts):
            if not text or conf < 0.65:
                continue
            stripped = text.strip()
            # Single punctuation character is almost always a misread dot or comma.
            if len(stripped) == 1 and not stripped.isalnum():
                continue
            results.append({"text": stripped, "conf": conf, "box": box})
        return results

    def _detect(self, img_rgb: np.ndarray) -> List[np.ndarray]:
        h, w = img_rgb.shape[:2]
        scale_x = self.DET_W / w
        new_h   = max(32, int(math.ceil(h * scale_x / 32) * 32))
        resized = cv2.resize(img_rgb, (self.DET_W, new_h))
        norm    = (resized.astype(np.float32) / 255.0 - self.DET_MEAN) / self.DET_STD
        tensor  = np.transpose(norm, (2, 0, 1))[np.newaxis]

        in_name  = self._det_sess.get_inputs()[0].name
        out_name = self._det_sess.get_outputs()[0].name
        prob_map = self._det_sess.run([out_name], {in_name: tensor})[0][0, 0]

        binary  = (prob_map > 0.25).astype(np.uint8) * 255
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)  # bridge thin-stroke gaps
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scale_y = new_h / h
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < 20:  # low threshold preserves thin strokes (l, i, 1)
                continue
            rect  = cv2.minAreaRect(c)
            pts   = cv2.boxPoints(rect).astype(np.float32)
            pts[:, 0] /= scale_x
            pts[:, 1] /= scale_y
            boxes.append(self._sort_corners(pts))
        return boxes

    @staticmethod
    def _sort_corners(box: np.ndarray) -> np.ndarray:
        s = box.sum(axis=1)
        d = np.diff(box, axis=1).squeeze()
        return np.array(
            [box[s.argmin()], box[d.argmin()], box[s.argmax()], box[d.argmax()]],
            dtype=np.float32,
        )

    def _recognize(
        self, img_rgb: np.ndarray, boxes: List[np.ndarray]
    ) -> List[Tuple[str, float]]:
        results = []
        for box in boxes:
            h_orig, w_orig = img_rgb.shape[:2]
            crop_w = max(
                int(np.linalg.norm(box[1] - box[0])),
                int(np.linalg.norm(box[2] - box[3])),
            )
            crop_h = max(
                int(np.linalg.norm(box[3] - box[0])),
                int(np.linalg.norm(box[2] - box[1])),
            )
            crop_w = max(crop_w, 1)
            crop_h = max(crop_h, 1)
            dst = np.array(
                [[0, 0], [crop_w - 1, 0], [crop_w - 1, crop_h - 1], [0, crop_h - 1]],
                dtype=np.float32,
            )
            M    = cv2.getPerspectiveTransform(box, dst)
            crop = cv2.warpPerspective(img_rgb, M, (crop_w, crop_h))

            scale  = self.REC_H / max(crop_h, 1)
            new_w  = max(1, int(round(crop_w * scale)))
            resized = cv2.resize(crop, (new_w, self.REC_H))
            norm    = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
            tensor  = np.transpose(norm, (2, 0, 1))[np.newaxis]

            in_name  = self._rec_sess.get_inputs()[0].name
            out_name = self._rec_sess.get_outputs()[0].name
            logits   = self._rec_sess.run([out_name], {in_name: tensor})[0][0]
            text, conf = self._ctc_decode(logits)
            results.append((text, conf))
        return results

    def _ctc_decode(self, logits: np.ndarray) -> Tuple[str, float]:
        indices = logits.argmax(axis=-1)
        probs   = logits.max(axis=-1)
        chars   = []
        conf_vals = []
        prev    = -1
        for idx, prob in zip(indices, probs):
            if idx != prev and idx != 0:
                if idx < len(self._vocab):
                    chars.append(self._vocab[idx])
                    conf_vals.append(float(prob))
            prev = idx
        text = "".join(chars)
        conf = float(np.mean(conf_vals)) if conf_vals else 0.0
        return text, conf


# ---------------------------------------------------------------------------
# DigitalWhiteboard
# ---------------------------------------------------------------------------
class DigitalWhiteboard:
    def __init__(self, width: int, height: int) -> None:
        self._w = width
        self._h = height
        self._img = PIL.Image.new("RGBA", (width, height), (255, 255, 255, 255))
        self._draw = PIL.ImageDraw.Draw(self._img)
        self._writer_colors: Dict[str, Tuple[int, int, int]] = {}
        self._color_idx = 0

    def get_writer_color(self, writer_id: str) -> Tuple[int, int, int]:
        if writer_id == "unknown":
            return UNKNOWN_COLOR
        if writer_id not in self._writer_colors:
            self._writer_colors[writer_id] = WRITER_PALETTE[
                self._color_idx % len(WRITER_PALETTE)
            ]
            self._color_idx += 1
        return self._writer_colors[writer_id]

    def add_text(
        self, text: str, box: np.ndarray, writer_id: str
    ) -> None:
        color = self.get_writer_color(writer_id)
        pts   = box.astype(np.int32)
        h_left  = float(np.linalg.norm(pts[3] - pts[0]))
        h_right = float(np.linalg.norm(pts[2] - pts[1]))
        font_sz = max(10, int((h_left + h_right) / 2.0 * 0.8))
        x, y = int(pts[0][0]), int(pts[0][1])
        try:
            font = PIL.ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_sz)
        except (OSError, IOError):
            font = PIL.ImageFont.load_default()
        self._draw.text((x, y), text, font=font, fill=color + (255,))

    def add_trace(
        self,
        contours: tuple,
        writer_id: str,
    ) -> None:
        color = self.get_writer_color(writer_id)
        overlay = PIL.Image.new("RGBA", (self._w, self._h), (0, 0, 0, 0))
        ov_draw = PIL.ImageDraw.Draw(overlay)
        for c in contours:
            if cv2.contourArea(c) < 200:
                continue
            pts = c.reshape(-1, 2).tolist()
            if len(pts) >= 3:
                flat = [coord for pt in pts for coord in pt]
                ov_draw.polygon(flat, fill=color + (180,), outline=color + (255,))
        self._img = PIL.Image.alpha_composite(self._img, overlay)
        self._draw = PIL.ImageDraw.Draw(self._img)

    def clear(self) -> None:
        self._img  = PIL.Image.new("RGBA", (self._w, self._h), (255, 255, 255, 255))
        self._draw = PIL.ImageDraw.Draw(self._img)

    def get_ctk_image(self, display_w: int, display_h: int) -> ctk.CTkImage:
        img_w, img_h = self._img.size
        scale = min(display_w / max(img_w, 1), display_h / max(img_h, 1))
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))
        resized = self._img.resize((new_w, new_h), PIL.Image.LANCZOS)
        canvas  = PIL.Image.new("RGBA", (display_w, display_h), (255, 255, 255, 255))
        ox = (display_w - new_w) // 2
        oy = (display_h - new_h) // 2
        canvas.paste(resized, (ox, oy))
        rgb = canvas.convert("RGB")
        return ctk.CTkImage(light_image=rgb, size=(int(display_w), int(display_h)))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
MIN_MOTION_VARIANCE        = 0.08   # minimum variance to pass attribution
MAX_MOTION_VARIANCE        = 5.0    # above this = handoff shaking, not writing — skip entirely
PROTOTYPE_UPDATE_VARIANCE  = 0.10   # higher bar — only update model with clear writing
ATTRIBUTION_WINDOW         = 10.0
ENROLL_STROKES             = 2      # strokes collected silently before comparison starts
BELOW_THRESH_STRIKES       = 2      # misses in MATCH_WINDOW → new writer (faster handoff detection)
MATCH_WINDOW               = 18      # rolling window for writer-switch detection
PEN_RESET_SECS             = 3.0    # seconds of pen silence → clear match window
PEN_WRITER_RESET_SECS      = 5.0    # longer silence → also clear writer identity (likely handoff)
MIN_PEN_IDLE_FOR_OCR       = 2.0    # seconds since last pen motion before OCR is processed
SOFT_REIDENTIFY_THRESHOLD  = 0.62   # before creating a new writer, soft-match against known prototypes
PEN_MOTION_THRESHOLD       = 0.8    # motion_score above this = pen is actively moving


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Whiteboard Digitizer")
        self.after(0, lambda: self.wm_state("zoomed"))

        self._imu_queue:     queue.Queue = queue.Queue(maxsize=500)
        self._display_queue: queue.Queue = queue.Queue(maxsize=2)
        self._cv_queue:      queue.Queue = queue.Queue(maxsize=4)

        self._writer_engine = WriterIdentityEngine()
        self._ocr            = PaddleOCRPipeline()
        self._ble            = BLEManager(self._imu_queue)
        self._segmenter      = Segmenter(min_active_samples=25)
        self._bulletin       = BulletinBoard()
        self._whiteboard: Optional[DigitalWhiteboard] = None
        self._cv_worker: Optional[CVWorker] = None
        self._vid: Optional[MyVideoCapture] = None
        self._pending_enroll: Optional[str] = None
        self._last_writer: Optional[str] = None
        self._ble_devices: List[bleak.BLEDevice] = []
        self._imu_player: Optional[CSVIMUPlayer] = None
        self._pending_recording: Optional[Tuple[Path, Path]] = None
        self._pen_reset_timer: Optional[str] = None
        self._writer_reset_timer: Optional[str] = None
        self._whiteboard_text_count: int = 0
        self._auto_writer_count = 0
        self._match_buffer: Deque[bool] = collections.deque(maxlen=MATCH_WINDOW)
        self._writer_stroke_counts: Dict[str, int] = {}
        self._last_stroke_time: float = 0.0
        self._last_motion_time: float = 0.0
        self._ocr_attributions: List[dict] = []
        self._bulletin_labels: List[ctk.CTkFrame] = []

        # camera list for selector
        self._camera_list = enumerate_cameras()

        self._build_ui()

        # ROI selector (needs canvas, created in _build_ui)
        self._roi_selector = ROISelector(self._webcam_canvas, self._on_roi_selected)

        self._imu_poll()
        self._display_poll()
        self._cv_poll()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ---- Sidebar ----
        self._sidebar = ctk.CTkFrame(self, width=340, corner_radius=0)
        self._sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self._sidebar.grid_propagate(False)
        self._sidebar.grid_rowconfigure(4, weight=1)

        # Controls
        ctrl = ctk.CTkFrame(self._sidebar)
        ctrl.grid(row=0, column=0, padx=10, pady=(10, 4), sticky="ew")
        ctk.CTkLabel(ctrl, text="Controls", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(8, 6))

        self._cam_menu = ctk.CTkOptionMenu(
            ctrl, values=[c.name for c in self._camera_list] or ["No cameras found"]
        )
        self._cam_menu.pack(padx=12, pady=4, fill="x")
        self._cam_btn = ctk.CTkButton(ctrl, text="Connect Camera", command=self._connect_camera)
        self._cam_btn.pack(padx=12, pady=4, fill="x")

        ctk.CTkButton(ctrl, text="Load Recording", command=self._open_recording_picker).pack(padx=12, pady=4, fill="x")

        self._roi_btn = ctk.CTkButton(ctrl, text="Select ROI (4 clicks)", command=self._toggle_roi)
        self._roi_btn.pack(padx=12, pady=4, fill="x")

        ctk.CTkButton(ctrl, text="Connect Pen", command=self._open_ble_popup).pack(padx=12, pady=4, fill="x")
        self._ble_status = ctk.CTkLabel(ctrl, text="Pen: disconnected", text_color="gray")
        self._ble_status.pack(padx=12, pady=(0, 8))

        # Enrollment
        enroll = ctk.CTkFrame(self._sidebar)
        enroll.grid(row=1, column=0, padx=10, pady=4, sticky="ew")
        ctk.CTkLabel(enroll, text="Writer ID", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(8, 4))
        self._name_entry = ctk.CTkEntry(enroll, placeholder_text="Name")
        self._name_entry.pack(padx=12, pady=4, fill="x")
        ctk.CTkButton(enroll, text="Enroll Next Stroke", command=self._arm_enrollment).pack(padx=12, pady=(4, 2), fill="x")
        ctk.CTkButton(enroll, text="Rename Last Writer", command=self._rename_last_writer,
                      fg_color="transparent", border_width=1).pack(padx=12, pady=(2, 4), fill="x")
        self._writer_label = ctk.CTkLabel(enroll, text="Writer: —")
        self._writer_label.pack(padx=12, pady=(0, 8))

        # Settings
        settings = ctk.CTkFrame(self._sidebar)
        settings.grid(row=2, column=0, padx=10, pady=4, sticky="ew")
        ctk.CTkLabel(settings, text="Settings", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(8, 4))
        self._gpu_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(settings, text="GPU Acceleration", variable=self._gpu_var, command=self._on_gpu_toggle).pack(padx=12, pady=(0, 8))


        # Bulletin
        bulletin_outer = ctk.CTkFrame(self._sidebar)
        bulletin_outer.grid(row=4, column=0, padx=10, pady=4, sticky="nsew")
        bulletin_outer.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(bulletin_outer, text="Timeline", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, pady=(8, 4), padx=12, sticky="w")
        self._bulletin_scroll = ctk.CTkScrollableFrame(bulletin_outer, label_text="")
        self._bulletin_scroll.grid(row=1, column=0, padx=4, pady=4, sticky="nsew")
        bulletin_outer.grid_columnconfigure(0, weight=1)

        # ---- Webcam canvas ----
        self._webcam_canvas = tk.Canvas(self, bg="#111")
        self._webcam_canvas.grid(row=0, column=1, padx=8, pady=8, sticky="nsew")

        # ---- Snapshot canvas ----
        self._snapshot_canvas = tk.Canvas(self, bg="#222")
        self._snapshot_canvas.grid(row=1, column=1, padx=8, pady=8, sticky="nsew")

        # ---- Digital twin (wrapper frame prevents image from expanding the column) ----
        self._twin_frame = ctk.CTkFrame(self, fg_color="#1a1a2e", corner_radius=8)
        self._twin_frame.grid(row=0, column=2, rowspan=2, padx=8, pady=8, sticky="nsew")
        self._twin_frame.grid_propagate(False)
        self._twin_label = ctk.CTkLabel(self._twin_frame, text="No ROI selected",
                                        fg_color="transparent")
        self._twin_label.place(relx=0.5, rely=0.5, anchor="center")

        # ---- ROI instruction label (hidden by default) ----
        self._roi_instruction = ctk.CTkLabel(
            self, text="Click top-left → top-right → bottom-right → bottom-left",
            fg_color="#e67e22", text_color="white",
        )

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def _connect_camera(self) -> None:
        self._pending_recording   = None
        self._auto_writer_count   = 0
        self._match_buffer.clear()
        self._last_writer         = None
        self._writer_stroke_counts = {}
        if self._pen_reset_timer is not None:
            self.after_cancel(self._pen_reset_timer)
            self._pen_reset_timer = None
        if self._writer_reset_timer is not None:
            self.after_cancel(self._writer_reset_timer)
            self._writer_reset_timer = None
        if self._writer_engine.registry:
            self._writer_engine.registry.prototypes.clear()
        if self._imu_player is not None:
            self._imu_player.stop()
            self._imu_player = None
            self._ble_status.configure(text="Pen: disconnected", text_color="gray")
        if self._cv_worker is not None:
            self._cv_worker.stop()

        selected = self._cam_menu.get()
        src = 0
        for i, dev in enumerate(self._camera_list):
            if dev.name == selected:
                src = i
                break
        try:
            self._vid = MyVideoCapture(src)
        except ValueError as exc:
            self._writer_label.configure(text=str(exc))
            return

        self._cam_btn.configure(text="Connect Camera")
        self._cv_worker = CVWorker(self._vid, self._display_queue, self._cv_queue)
        self._cv_worker.start()

    # ------------------------------------------------------------------
    # ROI
    # ------------------------------------------------------------------
    def _toggle_roi(self) -> None:
        if self._roi_selector._active:
            self._roi_selector.cancel()
            self._roi_btn.configure(text="Select ROI (4 clicks)")
            self._roi_instruction.place_forget()
        else:
            if self._vid is None:
                return
            self._roi_btn.configure(text="Cancel ROI selection")
            self._roi_instruction.place(relx=0.5, rely=0.01, anchor="n")
            self._roi_selector.start()

    def _on_roi_selected(self, corners: np.ndarray) -> None:
        self._roi_btn.configure(text="Reset ROI")
        self._roi_instruction.place_forget()

        if self._pending_recording is not None:
            video_path, csv_path = self._pending_recording
            self._pending_recording = None
            try:
                self._vid = FileVideoCapture(str(video_path))
            except ValueError as exc:
                self._writer_label.configure(text=str(exc))
                return
            self._cv_worker = CVWorker(self._vid, self._display_queue, self._cv_queue)
            self._cv_worker.set_roi(corners)
            self._whiteboard = DigitalWhiteboard(self._cv_worker._warp_w, self._cv_worker._warp_h)
            self._whiteboard_text_count = 0
            self._ocr_attributions = []
            self._twin_label.configure(text="")
            self._refresh_twin()
            self._cv_worker.start()
            self._imu_player = CSVIMUPlayer(str(csv_path), self._imu_queue)
            self._imu_player.start()
            label = self._imu_player.writer_label or "?"
            self._ble_status.configure(text=f"IMU: file {label}", text_color="#3a9bd5")
            self._cam_btn.configure(text=f"Live cam  (now: {video_path.stem})")
            return

        if self._cv_worker:
            self._cv_worker.set_roi(corners)
        w = self._cv_worker._warp_w if self._cv_worker else 960
        h = self._cv_worker._warp_h if self._cv_worker else 540
        self._whiteboard = DigitalWhiteboard(w, h)
        self._whiteboard_text_count = 0
        self._ocr_attributions = []
        self._twin_label.configure(text="")
        self._refresh_twin()

    # ------------------------------------------------------------------
    # BLE popup
    # ------------------------------------------------------------------
    def _open_ble_popup(self) -> None:
        popup = ctk.CTkToplevel(self)
        popup.title("Connect Pen")
        popup.geometry("400x300")
        popup.grab_set()

        status_lbl = ctk.CTkLabel(popup, text="Scanning… (10 s)", text_color="gray")
        status_lbl.pack(pady=(14, 4))

        device_var = ctk.StringVar(value="")
        dev_menu   = ctk.CTkOptionMenu(popup, values=["—"], variable=device_var, state="disabled")
        dev_menu.pack(padx=20, pady=6, fill="x")

        btn_row = ctk.CTkFrame(popup, fg_color="transparent")
        btn_row.pack(padx=20, pady=6, fill="x")

        rescan_btn  = ctk.CTkButton(btn_row, text="Rescan", width=100, state="disabled",
                                    command=lambda: _start_scan())
        rescan_btn.pack(side="left", padx=(0, 8))

        connect_btn = ctk.CTkButton(btn_row, text="Connect", state="disabled",
                                    command=lambda: _do_connect())
        connect_btn.pack(side="left", fill="x", expand=True)

        def _device_label(d: bleak.BLEDevice) -> str:
            return f"{d.name}  [{d.address}]" if d.name else d.address

        def _start_scan() -> None:
            rescan_btn.configure(state="disabled")
            connect_btn.configure(state="disabled")
            dev_menu.configure(state="disabled", values=["—"])
            status_lbl.configure(text="Scanning… (10 s)", text_color="gray")
            fut = self._ble.scan(timeout=10.0)

            def _poll():
                if not fut.done():
                    popup.after(300, _poll)
                    return
                try:
                    devices = fut.result()
                except Exception:
                    devices = []
                self._ble_devices = devices
                rescan_btn.configure(state="normal")
                if devices:
                    labels = [_device_label(d) for d in devices]
                    dev_menu.configure(values=labels, state="normal")
                    device_var.set(labels[0])
                    connect_btn.configure(state="normal")
                    status_lbl.configure(
                        text=f"Found {len(devices)} device(s)", text_color="gray"
                    )
                else:
                    status_lbl.configure(text="No devices found — try Rescan", text_color="orange")

            popup.after(300, _poll)

        def _do_connect() -> None:
            sel = device_var.get()
            device = next(
                (d for d in self._ble_devices if _device_label(d) == sel), None
            )
            if device is None:
                status_lbl.configure(text="Select a device first", text_color="orange")
                return

            connect_btn.configure(state="disabled", text="Connecting… (up to 30 s)")
            rescan_btn.configure(state="disabled")
            status_lbl.configure(text="Attempting connection (3 retries)…", text_color="gray")

            def _on_disconnect() -> None:
                self._ble_status.configure(text="Pen: disconnected", text_color="orange")

            fut = self._ble.connect(device, on_disconnect=_on_disconnect)

            def _poll():
                if not fut.done():
                    popup.after(300, _poll)
                    return
                try:
                    ok = fut.result()
                except Exception:
                    ok = False
                if ok:
                    if self._imu_player is not None:
                        self._imu_player.stop()
                        self._imu_player = None
                    label = device.name or device.address
                    self._ble_status.configure(text=f"Pen: {label}", text_color="green")
                    popup.destroy()
                else:
                    status_lbl.configure(
                        text="Connection failed — check device is on and try again",
                        text_color="red",
                    )
                    connect_btn.configure(state="normal", text="Connect")
                    rescan_btn.configure(state="normal")

            popup.after(300, _poll)

        _start_scan()

    # ------------------------------------------------------------------
    # IMU polling
    # ------------------------------------------------------------------
    def _imu_poll(self) -> None:
        try:
            while True:
                sample = self._imu_queue.get_nowait()
                if Segmenter.motion_score(sample) >= PEN_MOTION_THRESHOLD:
                    self._last_motion_time = time.time()
                seg    = self._segmenter.process_sample(sample)
                if seg is not None:
                    self._on_segment(seg)
        except queue.Empty:
            pass
        self.after(10, self._imu_poll)

    def _on_segment(self, segment: np.ndarray) -> None:
        scores = [Segmenter.motion_score(s) for s in segment]
        motion_var = float(np.var(scores))
        if motion_var < MIN_MOTION_VARIANCE or motion_var > MAX_MOTION_VARIANCE:
            return

        self._last_stroke_time = time.time()

        # Manual enrollment overrides everything.
        if self._pending_enroll is not None:
            writer_id = self._pending_enroll
            self._writer_engine.enroll_segment(writer_id, segment, momentum=0.5)
            self._pending_enroll = None
            self._last_writer = writer_id
            self._match_buffer.clear()
            color = self._whiteboard.get_writer_color(writer_id) if self._whiteboard else UNKNOWN_COLOR
            self._bulletin.post(BulletinEvent(kind="writer_enrolled", writer_id=writer_id))
            self._add_bulletin_row(f"Enrolled: {writer_id}", color)
            self._writer_label.configure(text=f"Enrolled: {writer_id}")
            return

        score = 0.0

        if self._last_writer is None:
            # Fresh start after a long idle (handoff). Resolve identity immediately on
            # the first stroke: soft-match against all known prototypes so a returning
            # participant is recognised without needing to accumulate misses.
            best_cand, best_cand_score = self._writer_engine.best_match_segment(segment)
            if best_cand is not None and best_cand_score >= SOFT_REIDENTIFY_THRESHOLD:
                writer_id = best_cand
                self._writer_engine.enroll_segment(writer_id, segment, momentum=0.85)
                score_str = f"re-id:{best_cand_score:.2f}"
            else:
                self._auto_writer_count += 1
                writer_id = f"Writer {self._auto_writer_count}"
                self._writer_engine.enroll_segment(writer_id, segment, momentum=0.5)
                self._writer_stroke_counts[writer_id] = 1
                score_str = "new"
            self._match_buffer.clear()
        else:
            current_enrolled = self._writer_stroke_counts.get(self._last_writer, 0)
            if current_enrolled < ENROLL_STROKES:
                # Silently build prototype before comparison begins.
                writer_id = self._last_writer
                if motion_var >= PROTOTYPE_UPDATE_VARIANCE:
                    self._writer_engine.enroll_segment(writer_id, segment, momentum=0.6)
                    self._writer_stroke_counts[writer_id] = current_enrolled + 1
                score_str = f"building ({self._writer_stroke_counts.get(writer_id, 1)}/{ENROLL_STROKES})"
            else:
                writer_id, score = self._writer_engine.predict_segment(segment)
                if writer_id is not None:
                    self._match_buffer.append(True)
                    if score >= 0.88 and motion_var >= PROTOTYPE_UPDATE_VARIANCE:
                        self._writer_engine.enroll_segment(writer_id, segment, momentum=0.95)
                    score_str = f"{score:.2f}"
                else:
                    self._match_buffer.append(False)
                    if self._match_buffer.count(False) >= BELOW_THRESH_STRIKES:
                        # Sliding-window fallback for short handoffs (<PEN_WRITER_RESET_SECS).
                        self._match_buffer.clear()
                        best_cand, best_cand_score = self._writer_engine.best_match_segment(segment)
                        if (best_cand is not None and
                                best_cand != self._last_writer and
                                best_cand_score >= SOFT_REIDENTIFY_THRESHOLD):
                            writer_id = best_cand
                            self._writer_engine.enroll_segment(writer_id, segment, momentum=0.85)
                            score_str = f"re-id:{best_cand_score:.2f}"
                        else:
                            self._auto_writer_count += 1
                            writer_id = f"Writer {self._auto_writer_count}"
                            self._writer_engine.enroll_segment(writer_id, segment, momentum=0.5)
                            self._writer_stroke_counts[writer_id] = 1
                            score_str = "new"
                    else:
                        writer_id = self._last_writer
                        score_str = f"~{score:.2f}"

        self._last_writer = writer_id
        self._bulletin.post(BulletinEvent(kind="pen_segment", writer_id=writer_id, sim_score=score))
        color = (self._whiteboard.get_writer_color(writer_id) if self._whiteboard else UNKNOWN_COLOR)
        ts_str = time.strftime("%H:%M:%S")
        self._add_bulletin_row(f"[{ts_str}] Pen: {writer_id} ({score_str})", color)
        self._writer_label.configure(text=f"Writer: {writer_id} ({score_str})")

        if self._pen_reset_timer is not None:
            self.after_cancel(self._pen_reset_timer)
        self._pen_reset_timer = self.after(int(PEN_RESET_SECS * 1000), self._on_pen_reset)
        if self._writer_reset_timer is not None:
            self.after_cancel(self._writer_reset_timer)
        self._writer_reset_timer = self.after(int(PEN_WRITER_RESET_SECS * 1000), self._on_writer_reset)

    # ------------------------------------------------------------------
    # CV polling
    # ------------------------------------------------------------------
    def _display_poll(self) -> None:
        try:
            while True:
                frame = self._display_queue.get_nowait()
                self._update_webcam_canvas(frame)
        except queue.Empty:
            pass
        self.after(15, self._display_poll)

    def _on_pen_reset(self) -> None:
        self._pen_reset_timer = None
        # Don't clear the match buffer here — let it slide naturally via deque maxlen.
        # Any clear here loses accumulated miss evidence (False entries) when P2 happens
        # to get a stray True reading between pauses.

    def _on_writer_reset(self) -> None:
        self._writer_reset_timer = None
        self._last_writer = None
        self._match_buffer.clear()

    def _cv_poll(self) -> None:
        # Drain the queue, keeping only the most recent result.
        latest = None
        try:
            while True:
                latest = self._cv_queue.get_nowait()
        except queue.Empty:
            pass
        pen_idle = time.time() - max(self._last_stroke_time, self._last_motion_time)
        if latest is not None and pen_idle >= MIN_PEN_IDLE_FOR_OCR:
            self._handle_cv_change(latest)
        self.after(50, self._cv_poll)

    def _update_webcam_canvas(self, frame: np.ndarray) -> None:
        cw = self._webcam_canvas.winfo_width()
        ch = self._webcam_canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        fh, fw = frame.shape[:2]
        self._roi_selector.update_frame_size(fw, fh, cw, ch)
        resized = cv2.resize(frame, (cw, ch))
        photo   = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(resized))
        self._webcam_canvas.delete("video")
        self._webcam_canvas.create_image(0, 0, image=photo, anchor=tk.NW, tags="video")
        self._webcam_canvas._photo_ref = photo  # prevent GC

    @staticmethod
    def _boxes_near(a: List, b: List, x_tol: float = 100.0, y_tol: float = 30.0) -> bool:
        # Elliptical tolerance: lenient in x (words grow rightward as writing continues),
        # strict in y (each line of text stays at its own row).
        ca = np.mean(np.array(a, dtype=np.float32), axis=0)
        cb = np.mean(np.array(b, dtype=np.float32), axis=0)
        return ((ca[0] - cb[0]) ** 2 / x_tol ** 2 +
                (ca[1] - cb[1]) ** 2 / y_tol ** 2) < 1.0

    def _handle_cv_change(self, result: dict) -> None:
        current_writer = self._bulletin.last_active_writer(time.time(), ATTRIBUTION_WINDOW) or "unknown"
        warped = result["warped"]
        self._update_snapshot_canvas(warped)

        ocr_results = self._ocr.run(warped)
        if not ocr_results:
            return

        # Additive model: text regions only accumulate — existing entries are never removed
        # because the writer's hand may be temporarily covering them. Updates are accepted
        # only when strictly better: higher confidence AND same-or-longer text (word grew).
        whiteboard_changed = False
        for item in ocr_results:
            matched_idx: Optional[int] = None
            for i, old in enumerate(self._ocr_attributions):
                if self._boxes_near(item["box"], old["box"]):
                    matched_idx = i
                    break

            if matched_idx is not None:
                old = self._ocr_attributions[matched_idx]
                new_text = item["text"]
                old_text = old["text"]
                # A genuine word-growth update shares its leading characters with the
                # stored text. "Shopping" growing from "Shop" passes; "hopping" replacing
                # "shop" (box shifted because the 'S' was missed) fails the prefix check.
                prefix_len = min(len(old_text), len(new_text), 3)
                same_prefix = (new_text[:prefix_len].lower() ==
                               old_text[:prefix_len].lower())
                # Accept if longer with same prefix (word grew), or same length with higher
                # confidence. Confidence alone is not required for longer text — a partial
                # read of a growing word often has artificially high confidence.
                longer_growth = len(new_text) > len(old_text) and same_prefix
                better_same_len = (len(new_text) == len(old_text) and same_prefix and
                                   item["conf"] > old.get("conf", 0.0))
                if longer_growth or better_same_len:
                    self._ocr_attributions[matched_idx] = {
                        "box": item["box"], "text": new_text,
                        "conf": item["conf"], "writer_id": old["writer_id"],
                    }
                    whiteboard_changed = True
            else:
                self._ocr_attributions.append({
                    "box": item["box"], "text": item["text"],
                    "conf": item["conf"], "writer_id": current_writer,
                })
                writer_id = current_writer
                color = (self._whiteboard.get_writer_color(writer_id) if self._whiteboard
                         else UNKNOWN_COLOR)
                ts_str = time.strftime("%H:%M:%S")
                self._bulletin.post(BulletinEvent(
                    kind="cv_update", cv_type="ocr",
                    writer_id=writer_id, text_content=item["text"],
                ))
                self._add_bulletin_row(
                    f'[{ts_str}] OCR ({writer_id}): "{item["text"][:40]}"', color
                )
                whiteboard_changed = True

        if whiteboard_changed and self._whiteboard:
            self._whiteboard.clear()
            for attr in self._ocr_attributions:
                self._whiteboard.add_text(attr["text"], attr["box"], attr["writer_id"])
            self._refresh_twin()

    def _update_snapshot_canvas(self, warped: np.ndarray) -> None:
        cw = self._snapshot_canvas.winfo_width()
        ch = self._snapshot_canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        resized = cv2.resize(warped, (cw, ch))
        photo   = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(resized))
        self._snapshot_canvas.delete("all")
        self._snapshot_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self._snapshot_canvas._photo_ref = photo

    def _refresh_twin(self) -> None:
        if self._whiteboard is None:
            return
        dw = self._twin_frame.winfo_width()
        dh = self._twin_frame.winfo_height()
        if dw < 2 or dh < 2:
            dw, dh = 640, 480
        ctk_img = self._whiteboard.get_ctk_image(dw, dh)
        self._twin_label.configure(image=ctk_img, text="")
        self._twin_label._ctk_img_ref = ctk_img

    # ------------------------------------------------------------------
    # Enrollment / rename
    # ------------------------------------------------------------------
    def _rename_last_writer(self) -> None:
        new_name = self._name_entry.get().strip()
        if not new_name:
            self._writer_label.configure(text="Enter a name first")
            return
        old_name = self._last_writer
        if old_name is None:
            self._writer_label.configure(text="No active writer to rename")
            return
        if old_name == new_name:
            return
        reg = self._writer_engine.registry
        if reg and old_name in reg.prototypes:
            reg.prototypes[new_name] = reg.prototypes.pop(old_name)
        if self._whiteboard and old_name in self._whiteboard._writer_colors:
            self._whiteboard._writer_colors[new_name] = self._whiteboard._writer_colors.pop(old_name)
        self._last_writer = new_name
        self._writer_label.configure(text=f"Renamed: {old_name} → {new_name}")

    def _arm_enrollment(self) -> None:
        name = self._name_entry.get().strip()
        if not name:
            self._writer_label.configure(text="Enter a name first")
            return
        self._pending_enroll = name
        self._writer_label.configure(text=f"Waiting to enroll: {name}")

    # ------------------------------------------------------------------
    # GPU toggle
    # ------------------------------------------------------------------
    def _on_gpu_toggle(self) -> None:
        provider = "CUDAExecutionProvider" if self._gpu_var.get() else "CPUExecutionProvider"
        self._writer_engine.set_provider(provider)
        self._ocr.set_provider(provider)

    # ------------------------------------------------------------------
    # Bulletin UI helper
    # ------------------------------------------------------------------
    def _add_bulletin_row(self, text: str, color: Tuple[int, int, int]) -> None:
        hex_color = "#{:02x}{:02x}{:02x}".format(*color)
        row = ctk.CTkFrame(self._bulletin_scroll, fg_color="transparent")
        row.pack(fill="x", padx=4, pady=1)

        indicator = ctk.CTkFrame(row, width=4, fg_color=hex_color, corner_radius=2)
        indicator.pack(side="left", fill="y", padx=(0, 4))

        lbl = ctk.CTkLabel(row, text=text, anchor="w", font=ctk.CTkFont(size=11))
        lbl.pack(side="left", fill="x", expand=True)

        self._bulletin_labels.append(row)
        if len(self._bulletin_labels) > 100:
            oldest = self._bulletin_labels.pop(0)
            oldest.destroy()

        # scroll to bottom
        self._bulletin_scroll._parent_canvas.yview_moveto(1.0)

    # ------------------------------------------------------------------
    # Recording playback
    # ------------------------------------------------------------------
    def _discover_recording_pairs(self) -> List[dict]:
        pairs = []
        for video in sorted(_MEDIA_DIR.glob("recording*.mp4")):
            prefix = video.stem  # e.g. "recording1"
            csvs = sorted(_MEDIA_DIR.glob(f"{prefix}_*.csv"))
            if not csvs:
                continue
            csv = csvs[0]
            parts = csv.stem.split("_")
            label = parts[1] if len(parts) > 1 else "?"
            pairs.append({"video": video, "csv": csv, "label": label,
                          "name": f"{prefix}  (writer: {label})"})
        return pairs

    def _open_recording_picker(self) -> None:
        pairs = self._discover_recording_pairs()
        if not pairs:
            self._writer_label.configure(text="No recordings found in onnx/")
            return

        popup = ctk.CTkToplevel(self)
        popup.title("Load Recording")
        popup.geometry("360x180")
        popup.grab_set()

        ctk.CTkLabel(popup, text="Select a recording pair:").pack(pady=(16, 4))

        names = [p["name"] for p in pairs]
        sel_var = ctk.StringVar(value=names[0])
        ctk.CTkOptionMenu(popup, values=names, variable=sel_var).pack(padx=20, pady=8, fill="x")

        def _do_load() -> None:
            chosen = next(p for p in pairs if p["name"] == sel_var.get())
            popup.destroy()
            self._load_recording(chosen["video"], chosen["csv"])

        ctk.CTkButton(popup, text="Load", command=_do_load).pack(padx=20, pady=8, fill="x")

    def _load_recording(self, video_path: Path, csv_path: Path) -> None:
        # Stop any existing sources
        if self._cv_worker is not None:
            self._cv_worker.stop()
            self._cv_worker = None
        if self._imu_player is not None:
            self._imu_player.stop()
            self._imu_player = None
        self._vid = None
        self._pending_recording = None
        self._whiteboard = None
        self._whiteboard_text_count = 0
        self._twin_label.configure(text="No ROI selected")
        self._auto_writer_count   = 0
        self._match_buffer.clear()
        self._last_writer         = None
        self._writer_stroke_counts = {}
        self._whiteboard_text_count = 0
        self._ocr_attributions = []
        if self._writer_engine.registry:
            self._writer_engine.registry.prototypes.clear()

        for q in (self._imu_queue, self._display_queue, self._cv_queue):
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break

        # Show first frame as a static preview so the user can select ROI
        # before any data starts flowing.
        _cap = cv2.VideoCapture(str(video_path))
        ok, frame = _cap.read()
        _cap.release()
        if ok:
            self._update_webcam_canvas(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Non-None sentinel so _toggle_roi doesn't bail before playback starts
        self._vid = True  # type: ignore[assignment]
        self._pending_recording = (video_path, csv_path)

        parts = csv_path.stem.split("_")
        label = parts[1] if len(parts) > 1 else "?"
        self._ble_status.configure(text=f"IMU: file  (writer hint: {label})", text_color="#3a9bd5")
        self._cam_btn.configure(text=f"Loaded: {video_path.stem}")
        self._roi_btn.configure(text="Select ROI to start")
        self._writer_label.configure(text="Select ROI to begin playback")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def on_closing(self) -> None:
        if self._cv_worker:
            self._cv_worker.stop()
        if self._imu_player:
            self._imu_player.stop()
        self._ble.disconnect()
        self._ble.stop()
        self.destroy()


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    app.mainloop()

