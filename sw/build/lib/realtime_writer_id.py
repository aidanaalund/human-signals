from __future__ import annotations

import argparse
import struct
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import serial

from writer_id_torch import WriterRegistry, load_model_bundle
from writer_id_onnx import ONNXWriterRegistry


PACKET_FMT = "<BBHIhhhhhh"
PACKET_SIZE = struct.calcsize(PACKET_FMT)


def read_exactly(ser: serial.Serial, n: int) -> Optional[bytes]:
    buf = b""
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def sync_header(ser: serial.Serial) -> bool:
    while True:
        b = ser.read(1)
        if not b:
            return False
        if b == b"\xAA":
            b2 = ser.read(1)
            if b2 == b"\x55":
                return True


def read_imu_sample(ser: serial.Serial) -> Optional[Tuple[int, np.ndarray]]:
    ok = sync_header(ser)
    if not ok:
        return None

    payload = read_exactly(ser, PACKET_SIZE - 2)
    if payload is None:
        return None

    try:
        _, _, count, _, gx_d10, gy_d10, gz_d10, ax_mg, ay_mg, az_mg = struct.unpack(
            PACKET_FMT, b"\xAA\x55" + payload
        )
    except struct.error:
        return None

    sample6 = np.array(
        [
            ax_mg / 1000.0,
            ay_mg / 1000.0,
            az_mg / 1000.0,
            gx_d10 / 10.0,
            gy_d10 / 10.0,
            gz_d10 / 10.0,
        ],
        dtype=np.float32,
    )
    return count, sample6


class Segmenter:
    def __init__(
        self,
        pre_roll: int = 8,
        start_threshold: float = 1.00,
        stop_threshold: float = 0.45,
        quiet_samples_to_stop: int = 5,
        min_active_samples: int = 18,
        max_active_samples: int = 90,
        cooldown_samples: int = 2,
    ):
        self.pre_roll = pre_roll
        self.start_threshold = start_threshold
        self.stop_threshold = stop_threshold
        self.quiet_samples_to_stop = quiet_samples_to_stop
        self.min_active_samples = min_active_samples
        self.max_active_samples = max_active_samples
        self.cooldown_samples = cooldown_samples

        self.pre_buffer = deque(maxlen=pre_roll)
        self.active_segment = []
        self.is_writing = False
        self.quiet_count = 0
        self.cooldown_count = 0

    @staticmethod
    def motion_score(sample6: np.ndarray) -> float:
        ax, ay, az, gx, gy, gz = sample6
        acc_mag = np.sqrt(ax * ax + ay * ay + az * az)
        gyro_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
        acc_motion = abs(acc_mag - 1.0)
        return float(3.0 * acc_motion + 0.03 * gyro_mag)

    def process_sample(self, sample6: np.ndarray) -> Optional[np.ndarray]:
        score = self.motion_score(sample6)

        if self.cooldown_count > 0:
            self.cooldown_count -= 1

        if not self.is_writing:
            self.pre_buffer.append(sample6)
            if self.cooldown_count == 0 and score >= self.start_threshold:
                self.is_writing = True
                self.active_segment = list(self.pre_buffer)
                self.quiet_count = 0
            return None

        self.active_segment.append(sample6)
        if score < self.stop_threshold:
            self.quiet_count += 1
        else:
            self.quiet_count = 0

        should_stop = (
            self.quiet_count >= self.quiet_samples_to_stop
            or len(self.active_segment) >= self.max_active_samples
        )
        if not should_stop:
            return None

        segment = np.asarray(self.active_segment, dtype=np.float32)

        self.is_writing = False
        self.active_segment = []
        self.pre_buffer.clear()
        self.cooldown_count = self.cooldown_samples
        self.quiet_count = 0

        if len(segment) < self.min_active_samples:
            return None
        return segment


def _parse_enroll_map(text: str) -> Dict[int, str]:
    """
    Example: "3:alice,6:bob" means segment #3 enrolls into alice.
    """
    mapping: Dict[int, str] = {}
    if not text.strip():
        return mapping
    for part in text.split(","):
        idx_str, writer_id = part.split(":", maxsplit=1)
        mapping[int(idx_str.strip())] = writer_id.strip()
    return mapping


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time writer identification for IMU stream")
    p.add_argument("--port", type=str, default="COM5") # TODO: hard coded
    p.add_argument("--baud", type=int, default=230400)
    p.add_argument("--run-dir", type=str, required=True, help="Path to writer_runs/<run>")
    p.add_argument("--registry", type=str, default="writer_runs/registry.json")
    p.add_argument("--unknown-threshold", type=float, default=0.72)
    p.add_argument("--backend", type=str, choices=["torch", "onnx"], default="torch")
    p.add_argument("--onnx-path", type=str, default="")
    p.add_argument("--onnx-provider", type=str, default="CPUExecutionProvider")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--enroll",
        type=str,
        default="",
        help="Enrollment map by segment index, e.g. '3:alice,7:bob'",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    registry_path = Path(args.registry)

    if args.backend == "torch":
        model, channel_mean, channel_std, _ = load_model_bundle(run_dir=run_dir, device=args.device)
        registry = WriterRegistry(
            model=model,
            channel_mean=channel_mean,
            channel_std=channel_std,
            target_len=96,
            unknown_threshold=args.unknown_threshold,
            device=args.device,
        )
    else:
        channel_mean = np.load(run_dir / "channel_mean.npy")
        channel_std = np.load(run_dir / "channel_std.npy")
        onnx_path = Path(args.onnx_path) if args.onnx_path else (run_dir / "writer_encoder.onnx")
        registry = ONNXWriterRegistry(
            onnx_path=onnx_path,
            channel_mean=channel_mean,
            channel_std=channel_std,
            target_len=96,
            unknown_threshold=args.unknown_threshold,
            providers=[args.onnx_provider],
        )

    if registry_path.exists():
        registry.load_registry(registry_path)
        print(f"Loaded registry with {len(registry.prototypes)} users from {registry_path}")
    else:
        print("No existing registry found. Start with enrollment.")

    segmenter = Segmenter()
    enroll_map = _parse_enroll_map(args.enroll)

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.5)
    except serial.SerialException as exc:
        print(f"Could not open serial port: {exc}")
        return

    time.sleep(2.0)
    ser.reset_input_buffer()
    print("Listening for strokes. Press Ctrl+C to stop.")
    if enroll_map:
        print(f"Enrollment plan active: {enroll_map}")

    seg_count = 0
    try:
        while True:
            pkt = read_imu_sample(ser)
            if pkt is None:
                continue
            _, sample6 = pkt

            segment = segmenter.process_sample(sample6)
            if segment is None:
                continue

            seg_count += 1

            if seg_count in enroll_map:
                writer_id = enroll_map[seg_count]
                registry.update_writer(writer_id=writer_id, segment=segment)
                print(f"seg={seg_count:04d} enrolled={writer_id} len={len(segment)}")
                continue

            pred_writer, sim = registry.predict_or_unknown(segment)
            if pred_writer is None:
                print(f"seg={seg_count:04d} writer=UNKNOWN sim={sim:.3f} len={len(segment)}")
            else:
                print(f"seg={seg_count:04d} writer={pred_writer} sim={sim:.3f} len={len(segment)}")
                registry.update_writer(writer_id=pred_writer, segment=segment)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ser.close()
        registry.save_registry(registry_path)
        print(f"Saved registry to {registry_path}")


if __name__ == "__main__":
    main()
