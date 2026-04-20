from __future__ import annotations

import asyncio
import math
import queue
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

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
        unknown_threshold: float = 0.82,
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
        self.registry = ONNXWriterRegistry(
            onnx_path=self._onnx_path,
            channel_mean=channel_mean,
            channel_std=channel_std,
            target_len=96,
            unknown_threshold=self._threshold,
            providers=[self._provider],
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

    def enroll_segment(self, writer_id: str, segment: np.ndarray) -> None:
        if not self._loaded:
            self.load()
        self.registry.update_writer(writer_id=writer_id, segment=segment)


# ---------------------------------------------------------------------------
# MyVideoCapture
# ---------------------------------------------------------------------------
class MyVideoCapture:
    def __init__(self, video_source: int = 0) -> None:
        self.vid = cv2.VideoCapture(video_source)
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

    def connect(self, device: bleak.BLEDevice):
        return asyncio.run_coroutine_threadsafe(self._connect_async(device), self._loop)

    def disconnect(self):
        return asyncio.run_coroutine_threadsafe(self._disconnect_async(), self._loop)

    def is_connected(self) -> bool:
        return self._connected.is_set()

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)

    # --- async internals ---

    async def _scan_async(self, timeout: float) -> List[bleak.BLEDevice]:
        devices = await bleak.BleakScanner.discover(timeout=timeout)
        return [d for d in devices if d.name]

    async def _connect_async(self, device: bleak.BLEDevice) -> bool:
        try:
            self._client = bleak.BleakClient(device)
            await self._client.connect()
            await self._client.start_notify(IMU_CHAR_UUID, self._notification_handler)
            self._connected.set()
            return True
        except Exception:
            self._connected.clear()
            return False

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
    FPS                  = 5
    OBSTRUCTION_THRESH   = 45.0
    CHANGE_THRESH        = 15.0
    WARP_WIDTH           = 960

    def __init__(
        self,
        vid: MyVideoCapture,
        result_queue: "queue.Queue[dict]",
    ) -> None:
        self._vid = vid
        self._q   = result_queue
        self._lock = threading.Lock()
        self._corners: Optional[np.ndarray] = None
        self._M:   Optional[np.ndarray] = None
        self._warp_w = self.WARP_WIDTH
        self._warp_h = self.WARP_WIDTH
        self._reference: Optional[np.ndarray] = None
        self._last_warped: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def set_roi(self, corners: np.ndarray) -> None:
        with self._lock:
            self._corners = corners
            self._M, self._warp_w, self._warp_h = self._compute_warp(corners)
            self._reference = None
            self._last_warped = None

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
        interval = 1.0 / self.FPS
        while self._running:
            t0 = time.monotonic()
            ok, frame = self._vid.get_frame()
            if ok and frame is not None:
                self._process(frame)
            elapsed = time.monotonic() - t0
            rem = interval - elapsed
            if rem > 0:
                time.sleep(rem)

    def _process(self, frame: np.ndarray) -> None:
        try:
            self._q.put_nowait({"type": "frame", "raw": frame})
        except queue.Full:
            pass

        with self._lock:
            M      = self._M
            warp_w = self._warp_w
            warp_h = self._warp_h

        if M is None:
            return

        warped = cv2.warpPerspective(frame, M, (warp_w, warp_h))

        if self._is_obstructed(warped):
            return

        if self._reference is None:
            self._reference = warped.copy()
            return

        if not self._has_changed(warped):
            return

        gray_new = cv2.cvtColor(warped,          cv2.COLOR_RGB2GRAY)
        gray_ref = cv2.cvtColor(self._reference, cv2.COLOR_RGB2GRAY)
        diff      = cv2.absdiff(gray_new, gray_ref)
        mask      = (diff > 30).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        is_text   = self._classify_change(contours, mask)

        self._reference = warped.copy()

        try:
            self._q.put_nowait({
                "type":     "changed",
                "warped":   warped,
                "is_text":  is_text,
                "contours": contours,
            })
        except queue.Full:
            pass

    def _is_obstructed(self, warped: np.ndarray) -> bool:
        if self._last_warped is None:
            self._last_warped = warped.copy()
            return False
        diff = cv2.absdiff(warped, self._last_warped).mean()
        self._last_warped = warped.copy()
        return float(diff) > self.OBSTRUCTION_THRESH

    def _has_changed(self, warped: np.ndarray) -> bool:
        gray_new = cv2.cvtColor(warped,          cv2.COLOR_RGB2GRAY)
        gray_ref = cv2.cvtColor(self._reference, cv2.COLOR_RGB2GRAY)
        return float(cv2.absdiff(gray_new, gray_ref).mean()) > self.CHANGE_THRESH

    @staticmethod
    def _classify_change(
        contours: tuple, mask: np.ndarray
    ) -> bool:
        total_area = mask.shape[0] * mask.shape[1]
        text_like = 0
        significant = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 50:
                continue
            significant += 1
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / max(h, 1)
            if aspect > 2.0 and area < 0.05 * total_area:
                text_like += 1
        if significant == 0:
            return False
        return text_like / significant > 0.5


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

    def run(self, img_rgb: np.ndarray) -> List[Dict]:
        self._ensure_loaded()
        boxes = self._detect(img_rgb)
        if not boxes:
            return []
        texts = self._recognize(img_rgb, boxes)
        results = []
        for box, (text, conf) in zip(boxes, texts):
            if text:
                results.append({"text": text, "conf": conf, "box": box})
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

        binary  = (prob_map > 0.3).astype(np.uint8) * 255
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scale_y = new_h / h
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < 100:
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
MIN_MOTION_VARIANCE = 0.05
ATTRIBUTION_WINDOW  = 10.0


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Whiteboard Digitizer")
        self.after(0, lambda: self.wm_state("zoomed"))

        self._imu_queue: queue.Queue = queue.Queue(maxsize=500)
        self._cv_queue:  queue.Queue = queue.Queue(maxsize=20)

        self._writer_engine = WriterIdentityEngine()
        self._ocr            = PaddleOCRPipeline()
        self._ble            = BLEManager(self._imu_queue)
        self._segmenter      = Segmenter(min_active_samples=18)
        self._bulletin       = BulletinBoard()
        self._whiteboard: Optional[DigitalWhiteboard] = None
        self._cv_worker: Optional[CVWorker] = None
        self._vid: Optional[MyVideoCapture] = None
        self._pending_enroll: Optional[str] = None
        self._last_writer: Optional[str] = None
        self._ble_devices: List[bleak.BLEDevice] = []
        self._bulletin_labels: List[ctk.CTkFrame] = []

        # camera list for selector
        self._camera_list = enumerate_cameras()

        self._build_ui()

        # ROI selector (needs canvas, created in _build_ui)
        self._roi_selector = ROISelector(self._webcam_canvas, self._on_roi_selected)

        self._imu_poll()
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
        ctk.CTkButton(ctrl, text="Connect Camera", command=self._connect_camera).pack(padx=12, pady=4, fill="x")

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
        ctk.CTkButton(enroll, text="Enroll Next Stroke", command=self._arm_enrollment).pack(padx=12, pady=4, fill="x")
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

        # ---- Digital twin label ----
        self._twin_label = ctk.CTkLabel(self, text="No ROI selected", fg_color="#1a1a2e")
        self._twin_label.grid(row=0, column=2, rowspan=2, padx=8, pady=8, sticky="nsew")

        # ---- ROI instruction label (hidden by default) ----
        self._roi_instruction = ctk.CTkLabel(
            self, text="Click top-left → top-right → bottom-right → bottom-left",
            fg_color="#e67e22", text_color="white",
        )

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def _connect_camera(self) -> None:
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

        self._cv_worker = CVWorker(self._vid, self._cv_queue)
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
        if self._cv_worker:
            self._cv_worker.set_roi(corners)
        w = self._cv_worker._warp_w if self._cv_worker else 960
        h = self._cv_worker._warp_h if self._cv_worker else 540
        self._whiteboard = DigitalWhiteboard(w, h)
        self._twin_label.configure(text="")
        self._refresh_twin()

    # ------------------------------------------------------------------
    # BLE popup
    # ------------------------------------------------------------------
    def _open_ble_popup(self) -> None:
        popup = ctk.CTkToplevel(self)
        popup.title("Connect Pen")
        popup.geometry("360x260")
        popup.grab_set()

        ctk.CTkLabel(popup, text="Scanning for BLE devices…").pack(pady=12)
        status_lbl = ctk.CTkLabel(popup, text="", text_color="gray")
        status_lbl.pack()

        device_var = ctk.StringVar(value="")
        dev_menu   = ctk.CTkOptionMenu(popup, values=["Scanning…"], variable=device_var)
        dev_menu.pack(padx=20, pady=8, fill="x")
        dev_menu.configure(state="disabled")

        connect_btn = ctk.CTkButton(popup, text="Connect", state="disabled",
                                    command=lambda: self._ble_connect(popup, status_lbl, connect_btn))
        connect_btn.pack(padx=20, pady=8, fill="x")

        scan_fut = self._ble.scan(timeout=5.0)

        def poll_scan():
            if not scan_fut.done():
                popup.after(200, poll_scan)
                return
            devices = scan_fut.result()
            self._ble_devices = devices
            if devices:
                names = [d.name or d.address for d in devices]
                dev_menu.configure(values=names, state="normal")
                device_var.set(names[0])
                connect_btn.configure(state="normal")
                status_lbl.configure(text=f"Found {len(devices)} device(s)")
            else:
                status_lbl.configure(text="No devices found")

        popup.after(200, poll_scan)

    def _ble_connect(
        self,
        popup: ctk.CTkToplevel,
        status_lbl: ctk.CTkLabel,
        btn: ctk.CTkButton,
    ) -> None:
        sel_name = popup.children.get("!ctkoptionmenu")
        # Find the selected device by matching option menu value
        sel_val = None
        for child in popup.winfo_children():
            if isinstance(child, ctk.CTkOptionMenu):
                sel_val = child.get()
                break
        device = next(
            (d for d in self._ble_devices if (d.name or d.address) == sel_val), None
        )
        if device is None:
            status_lbl.configure(text="Device not found")
            return
        btn.configure(state="disabled", text="Connecting…")
        fut = self._ble.connect(device)

        def poll_connect():
            if not fut.done():
                popup.after(200, poll_connect)
                return
            ok = fut.result()
            if ok:
                self._ble_status.configure(text=f"Pen: {device.name}", text_color="green")
                popup.destroy()
            else:
                status_lbl.configure(text="Connection failed")
                btn.configure(state="normal", text="Connect")

        popup.after(200, poll_connect)

    # ------------------------------------------------------------------
    # IMU polling
    # ------------------------------------------------------------------
    def _imu_poll(self) -> None:
        try:
            while True:
                sample = self._imu_queue.get_nowait()
                seg    = self._segmenter.process_sample(sample)
                if seg is not None:
                    self._on_segment(seg)
        except queue.Empty:
            pass
        self.after(10, self._imu_poll)

    def _on_segment(self, segment: np.ndarray) -> None:
        scores = [Segmenter.motion_score(s) for s in segment]
        if float(np.var(scores)) < MIN_MOTION_VARIANCE:
            return

        if self._pending_enroll is not None:
            writer_id = self._pending_enroll
            self._writer_engine.enroll_segment(writer_id, segment)
            self._pending_enroll = None
            color = self._whiteboard.get_writer_color(writer_id) if self._whiteboard else UNKNOWN_COLOR
            self._bulletin.post(BulletinEvent(kind="writer_enrolled", writer_id=writer_id))
            self._add_bulletin_row(f"Enrolled: {writer_id}", color)
            self._writer_label.configure(text=f"Enrolled: {writer_id}")
            return

        writer_id, score = self._writer_engine.predict_segment(segment)
        if writer_id is not None:
            self._writer_engine.enroll_segment(writer_id, segment)
            self._last_writer = writer_id
        self._bulletin.post(BulletinEvent(
            kind="pen_segment",
            writer_id=writer_id or "unknown",
            sim_score=score,
        ))
        display = writer_id or "unknown"
        color   = (self._whiteboard.get_writer_color(display) if self._whiteboard
                   else UNKNOWN_COLOR)
        ts_str  = time.strftime("%H:%M:%S")
        self._add_bulletin_row(f"[{ts_str}] Pen: {display} ({score:.2f})", color)
        self._writer_label.configure(text=f"Writer: {display} ({score:.2f})")

    # ------------------------------------------------------------------
    # CV polling
    # ------------------------------------------------------------------
    def _cv_poll(self) -> None:
        try:
            while True:
                result = self._cv_queue.get_nowait()
                if result["type"] == "frame":
                    self._update_webcam_canvas(result["raw"])
                elif result["type"] == "changed":
                    self._handle_cv_change(result)
        except queue.Empty:
            pass
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

    def _handle_cv_change(self, result: dict) -> None:
        writer_id = self._bulletin.last_active_writer(time.time(), ATTRIBUTION_WINDOW) or "unknown"
        warped    = result["warped"]

        if result["is_text"]:
            ocr_results = self._ocr.run(warped)
            for item in ocr_results:
                if self._whiteboard:
                    self._whiteboard.add_text(item["text"], item["box"], writer_id)
                ts_str = time.strftime("%H:%M:%S")
                color  = (self._whiteboard.get_writer_color(writer_id) if self._whiteboard
                          else UNKNOWN_COLOR)
                snippet = item["text"][:40]
                self._bulletin.post(BulletinEvent(
                    kind="cv_update", cv_type="ocr",
                    writer_id=writer_id, text_content=item["text"],
                ))
                self._add_bulletin_row(f'[{ts_str}] OCR ({writer_id}): "{snippet}"', color)
        else:
            if self._whiteboard:
                self._whiteboard.add_trace(result["contours"], writer_id)
            color  = (self._whiteboard.get_writer_color(writer_id) if self._whiteboard
                      else UNKNOWN_COLOR)
            ts_str = time.strftime("%H:%M:%S")
            self._bulletin.post(BulletinEvent(
                kind="cv_update", cv_type="trace", writer_id=writer_id,
            ))
            self._add_bulletin_row(f"[{ts_str}] Trace ({writer_id})", color)

        if warped is not None:
            self._update_snapshot_canvas(warped)

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
        dw = self._twin_label.winfo_width()
        dh = self._twin_label.winfo_height()
        if dw < 2 or dh < 2:
            dw, dh = 640, 480
        ctk_img = self._whiteboard.get_ctk_image(dw, dh)
        self._twin_label.configure(image=ctk_img)
        self._twin_label._ctk_img_ref = ctk_img

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------
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
    # Cleanup
    # ------------------------------------------------------------------
    def on_closing(self) -> None:
        if self._cv_worker:
            self._cv_worker.stop()
        self._ble.disconnect()
        self._ble.stop()
        self.destroy()


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

