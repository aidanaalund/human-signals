import time
import struct
import threading
import asyncio
import queue
from collections import deque
from pathlib import Path

import numpy as np
from tensorflow import keras
from bleak import BleakClient, BleakScanner

from pynput import keyboard


# =========================
# BLE settings
# =========================
DEVICE_NAME = "MG24_IMU"
CHAR_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"

PACKET_FMT = "<BBHIhhhhhh"
PACKET_SIZE = struct.calcsize(PACKET_FMT)


# =========================
# Model artifacts
# =========================
# RUN_DIR = Path("model_runs") / "78_acc_cnn_run_20260312_003648"
RUN_DIR = Path("BLE_model_runs") / "98_acc_128len_cnnbilstm_run_20260413_111857"
MODEL_PATH = RUN_DIR / "imu_char_cnn.keras"
MEAN_PATH = RUN_DIR / "channel_mean.npy"
STD_PATH = RUN_DIR / "channel_std.npy"
CLASSES_PATH = RUN_DIR / "label_classes.npy"

TARGET_LEN = 128


# =========================
# Segmentation / inference tuning
# Adjusted from ~52 Hz to ~104 Hz
# =========================
START_THRESHOLD = 1.00
STOP_THRESHOLD = 0.45
QUIET_SAMPLES_TO_STOP = 10
MIN_ACTIVE_SAMPLES = 36
MAX_ACTIVE_SAMPLES = 140
CONF_THRESHOLD = 0.70
PRE_ROLL = 16
COOLDOWN_SAMPLES = 4

PRINT_SCORES = False
SILENT_TEXT_MODE = False

WINDOW_SIZE = 160
WINDOW_OVERLAP = 120

if WINDOW_OVERLAP >= WINDOW_SIZE:
    raise ValueError("WINDOW_OVERLAP must be smaller than WINDOW_SIZE")


def parse_ble_packet(data: bytearray):
    if len(data) != PACKET_SIZE:
        return None

    try:
        h1, h2, count, t_us, gx_d10, gy_d10, gz_d10, ax_mg, ay_mg, az_mg = struct.unpack(
            PACKET_FMT, data
        )
    except struct.error:
        return None

    if h1 != 0xAA or h2 != 0x55:
        return None

    gx = gx_d10 / 10.0
    gy = gy_d10 / 10.0
    gz = gz_d10 / 10.0

    ax = ax_mg / 1000.0
    ay = ay_mg / 1000.0
    az = az_mg / 1000.0

    sample6 = np.array([ax, ay, az, gx, gy, gz], dtype=np.float32)
    return count, t_us, sample6


# =========================
# Preprocessing (must match training)
# =========================
def preprocess_sample(sample, center=True, add_magnitude=True):
    """
    sample shape: (T, 6)
    columns: ax, ay, az, gx, gy, gz
    """
    x = sample.astype(np.float32).copy()

    if center:
        x[:, :3] -= x[:, :3].mean(axis=0, keepdims=True)
        x[:, 3:6] -= x[:, 3:6].mean(axis=0, keepdims=True)

    if add_magnitude:
        acc_mag = np.linalg.norm(x[:, :3], axis=1, keepdims=True)
        gyro_mag = np.linalg.norm(x[:, 3:6], axis=1, keepdims=True)
        x = np.hstack([x, acc_mag, gyro_mag])

    return x


def resample_sequence(seq, target_len=64):
    """
    seq: (T, C) -> (target_len, C)
    """
    T, C = seq.shape
    if T == target_len:
        return seq.astype(np.float32)

    old_idx = np.linspace(0.0, 1.0, T)
    new_idx = np.linspace(0.0, 1.0, target_len)

    out = np.zeros((target_len, C), dtype=np.float32)
    for c in range(C):
        out[:, c] = np.interp(new_idx, old_idx, seq[:, c])

    return out


# =========================
# Recognizer
# =========================
class RealTimeCharRecognizer:
    def __init__(
        self,
        model,
        channel_mean,
        channel_std,
        label_classes,
        target_len=64,
        pre_roll=6,
        start_threshold=1.2,
        stop_threshold=0.35,
        quiet_samples_to_stop=6,
        min_active_samples=18,
        max_active_samples=100,
        conf_threshold=0.65,
        cooldown_samples=8,
    ):
        self.model = model
        self.channel_mean = channel_mean.reshape(1, 1, -1).astype(np.float32)
        self.channel_std = channel_std.reshape(1, 1, -1).astype(np.float32)
        self.label_classes = np.array(label_classes)

        self.target_len = target_len
        self.pre_roll = pre_roll
        self.start_threshold = start_threshold
        self.stop_threshold = stop_threshold
        self.quiet_samples_to_stop = quiet_samples_to_stop
        self.min_active_samples = min_active_samples
        self.max_active_samples = max_active_samples
        self.conf_threshold = conf_threshold
        self.cooldown_samples = cooldown_samples

        self.pre_buffer = deque(maxlen=pre_roll)
        self.active_segment = []
        self.is_writing = False
        self.quiet_count = 0
        self.cooldown_count = 0

    def motion_score(self, sample6):
        """
        sample6 = [ax, ay, az, gx, gy, gz]
        accel in g, gyro in dps
        """
        ax, ay, az, gx, gy, gz = sample6

        acc_mag = np.sqrt(ax * ax + ay * ay + az * az)
        gyro_mag = np.sqrt(gx * gx + gy * gy + gz * gz)

        acc_motion = abs(acc_mag - 1.0)
        score = 3.0 * acc_motion + 0.03 * gyro_mag
        return float(score)

    def prepare_for_model(self, raw_segment):
        x = preprocess_sample(raw_segment, center=True, add_magnitude=True)
        x = resample_sequence(x, self.target_len)
        x = x[np.newaxis, :, :]
        x = (x - self.channel_mean) / (self.channel_std + 1e-8)
        return x.astype(np.float32)

    def predict_segment(self, raw_segment):
        x = self.prepare_for_model(raw_segment)
        probs = self.model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = str(self.label_classes[pred_idx])
        conf = float(probs[pred_idx])
        return pred_label, conf, probs

    def process_sample(self, sample6):
        """
        Returns None or (pred_label, conf, seg_len)
        """
        score = self.motion_score(sample6)

        if PRINT_SCORES:
            print(f"score={score:.3f}")

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

        pred_label, conf, _ = self.predict_segment(segment)

        if conf < self.conf_threshold:
            return None

        return pred_label, conf, len(segment)


class BLEPacketReceiver:
    def __init__(self, packet_queue: queue.Queue, stop_event: threading.Event):
        self.packet_queue = packet_queue
        self.stop_event = stop_event
        self.thread = None
        self.loop = None

    def start(self):
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()

    def join(self, timeout=None):
        if self.thread is not None:
            self.thread.join(timeout=timeout)

    def _thread_main(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._ble_main())
        finally:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            try:
                self.loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            except Exception:
                pass
            self.loop.close()
            self.loop = None

    async def _find_device(self):
        devices = await BleakScanner.discover(timeout=5.0)
        for d in devices:
            if d.name == DEVICE_NAME:
                return d
        return None

    def _notification_handler(self, sender, data: bytearray):
        parsed = parse_ble_packet(data)
        if parsed is None:
            return
        try:
            self.packet_queue.put_nowait(parsed)
        except queue.Full:
            pass

    async def _ble_main(self):
        while not self.stop_event.is_set():
            try:
                print("Scanning for BLE device...")
                device = await self._find_device()

                if device is None:
                    print("Device not found; retrying...")
                    await asyncio.sleep(2.0)
                    continue

                print(f"Connecting to {device.name}...")

                async with BleakClient(device) as client:
                    print(f"Connected to {device.name}")
                    await client.start_notify(CHAR_UUID, self._notification_handler)

                    while client.is_connected and not self.stop_event.is_set():
                        await asyncio.sleep(0.2)

                    try:
                        await client.stop_notify(CHAR_UUID)
                    except Exception:
                        pass

            except Exception as e:
                print(f"BLE error: {e}")

            if not self.stop_event.is_set():
                await asyncio.sleep(2.0)


def start_keyboard_listener():
    def on_press(key):
        try:
            if key == keyboard.Key.enter:
                print()  # newline
        except Exception:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

def main():
    model = keras.models.load_model(MODEL_PATH)
    channel_mean = np.load(MEAN_PATH)
    channel_std = np.load(STD_PATH)
    label_classes = np.load(CLASSES_PATH, allow_pickle=True)

    recognizer = RealTimeCharRecognizer(
        model=model,
        channel_mean=channel_mean,
        channel_std=channel_std,
        label_classes=label_classes,
        target_len=TARGET_LEN,
        pre_roll=PRE_ROLL,
        start_threshold=START_THRESHOLD,
        stop_threshold=STOP_THRESHOLD,
        quiet_samples_to_stop=QUIET_SAMPLES_TO_STOP,
        min_active_samples=MIN_ACTIVE_SAMPLES,
        max_active_samples=MAX_ACTIVE_SAMPLES,
        conf_threshold=CONF_THRESHOLD,
        cooldown_samples=COOLDOWN_SAMPLES,
    )

    packet_queue = queue.Queue(maxsize=4096)
    stop_event = threading.Event()

    receiver = BLEPacketReceiver(packet_queue, stop_event)
    receiver.start()

    print("Listening over BLE... write a character. Ctrl+C to stop.")

    last_count = None
    drop_count = 0

    report_t0 = None
    report_count0 = None

    start_keyboard_listener()

    try:
        while True:
            try:
                count, t_us, sample6 = packet_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            now = time.perf_counter()

            if last_count is not None:
                expected = (last_count + 1) & 0xFFFF
                if count != expected:
                    delta = (count - expected) & 0xFFFF
                    drop_count += delta
                    print(f"[warn] dropped packets total={drop_count}")

            last_count = count

            if report_t0 is None:
                report_t0 = now
                report_count0 = count
            else:
                dt = now - report_t0
                if dt >= 2.0:
                    dc = (count - report_count0) & 0xFFFF
                    rate = dc / dt if dt > 0 else 0.0
                    if not SILENT_TEXT_MODE:
                        print(f"[rate] {rate:.2f} Hz | last_count={count} | dropped={drop_count}")
                    report_t0 = now
                    report_count0 = count

            result = recognizer.process_sample(sample6)
            if result is not None:
                pred_label, conf, seg_len = result

                if SILENT_TEXT_MODE:
                    print(pred_label, end="", flush=True)
                else:
                    print(f"{pred_label}    conf={conf:.3f}    len={seg_len}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stop_event.set()
        receiver.join(timeout=1.0)


if __name__ == "__main__":
    main()