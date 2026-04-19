import time
import struct
from collections import deque, Counter
from pathlib import Path

import numpy as np
import serial
from tensorflow import keras


# =========================
# Serial settings
# =========================
PORT = "COM5"
BAUD = 230400

PACKET_FMT = "<BBHIhhhhhh"
PACKET_SIZE = struct.calcsize(PACKET_FMT)


# =========================
# Model artifacts
# =========================
RUN_DIR = Path("model_runs") / "90_acc_cnnbilstm_run_20260312_014301"   # change this
MODEL_PATH = RUN_DIR / "imu_char_cnn.keras"
MEAN_PATH = RUN_DIR / "channel_mean.npy"
STD_PATH = RUN_DIR / "channel_std.npy"
CLASSES_PATH = RUN_DIR / "label_classes.npy"

TARGET_LEN = 64


# =========================
# Fixed-window params
# =========================
WINDOW_SIZE = 48
WINDOW_OVERLAP = 32   # hop = 8
CONF_THRESHOLD = 0.75

# print control
REPEAT_SUPPRESSION = True
STABLE_VOTES_REQUIRED = 3     # same label must appear this many times in a row
PRINT_TOPK = 0                # for debugging
PRINT_EVERY_WINDOW = False    # if True, prints every confident window


if WINDOW_OVERLAP >= WINDOW_SIZE:
    raise ValueError("WINDOW_OVERLAP must be smaller than WINDOW_SIZE")

HOP_SIZE = WINDOW_SIZE - WINDOW_OVERLAP


# =========================
# Serial helpers
# =========================
def read_exactly(ser, n):
    buf = b""
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def sync_header(ser):
    while True:
        b = ser.read(1)
        if not b:
            return False
        if b == b"\xAA":
            b2 = ser.read(1)
            if b2 == b"\x55":
                return True


def read_imu_sample(ser):
    ok = sync_header(ser)
    if not ok:
        return None

    payload = read_exactly(ser, PACKET_SIZE - 2)
    if payload is None:
        return None

    try:
        _, _, count, t_us, gx_d10, gy_d10, gz_d10, ax_mg, ay_mg, az_mg = struct.unpack(
            PACKET_FMT, b"\xAA\x55" + payload
        )
    except struct.error:
        return None

    gx = gx_d10 / 10.0
    gy = gy_d10 / 10.0
    gz = gz_d10 / 10.0

    ax = ax_mg / 1000.0
    ay = ay_mg / 1000.0
    az = az_mg / 1000.0

    sample6 = np.array([ax, ay, az, gx, gy, gz], dtype=np.float32)
    return count, sample6


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
# Fixed-window recognizer
# =========================
class FixedWindowRecognizer:
    def __init__(
        self,
        model,
        channel_mean,
        channel_std,
        label_classes,
        target_len=64,
        window_size=64,
        hop_size=8,
        conf_threshold=0.75,
        stable_votes_required=3,
        repeat_suppression=True,
        print_topk=3,
        print_every_window=False,
    ):
        self.model = model
        self.channel_mean = channel_mean.reshape(1, 1, -1).astype(np.float32)
        self.channel_std = channel_std.reshape(1, 1, -1).astype(np.float32)
        self.label_classes = np.array(label_classes)

        self.target_len = target_len
        self.window_size = window_size
        self.hop_size = hop_size
        self.conf_threshold = conf_threshold
        self.stable_votes_required = stable_votes_required
        self.repeat_suppression = repeat_suppression
        self.print_topk = print_topk
        self.print_every_window = print_every_window

        self.buffer = deque(maxlen=window_size)
        self.samples_since_last_infer = 0

        self.recent_labels = deque(maxlen=stable_votes_required)
        self.last_printed_label = None

    def prepare_for_model(self, raw_window):
        x = preprocess_sample(raw_window, center=True, add_magnitude=True)
        x = resample_sequence(x, self.target_len)
        x = x[np.newaxis, :, :]
        x = (x - self.channel_mean) / (self.channel_std + 1e-8)
        return x.astype(np.float32)

    def predict_window(self, raw_window):
        x = self.prepare_for_model(raw_window)
        probs = self.model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = str(self.label_classes[pred_idx])
        conf = float(probs[pred_idx])

        topk_idx = np.argsort(probs)[::-1][:self.print_topk]
        topk = [(str(self.label_classes[i]), float(probs[i])) for i in topk_idx]

        return pred_label, conf, probs, topk

    def process_sample(self, sample6):
        self.buffer.append(sample6)
        self.samples_since_last_infer += 1

        if len(self.buffer) < self.window_size:
            return None

        if self.samples_since_last_infer < self.hop_size:
            return None

        self.samples_since_last_infer = 0

        raw_window = np.asarray(self.buffer, dtype=np.float32)
        pred_label, conf, probs, topk = self.predict_window(raw_window)

        if self.print_every_window:
            print(f"window pred={pred_label} conf={conf:.3f} topk={topk}")

        if conf < self.conf_threshold:
            self.recent_labels.clear()
            return None

        self.recent_labels.append(pred_label)

        if len(self.recent_labels) < self.stable_votes_required:
            return None

        # require same confident label for K consecutive inference windows
        if len(set(self.recent_labels)) != 1:
            return None

        stable_label = self.recent_labels[-1]

        if self.repeat_suppression and stable_label == self.last_printed_label:
            return None

        self.last_printed_label = stable_label
        return stable_label, conf, topk

    def reset_print_latch(self):
        """
        Optional external reset if you want to allow the same character
        to be printed again later.
        """
        self.last_printed_label = None


def main():
    model = keras.models.load_model(MODEL_PATH)
    channel_mean = np.load(MEAN_PATH)
    channel_std = np.load(STD_PATH)
    label_classes = np.load(CLASSES_PATH, allow_pickle=True)

    recognizer = FixedWindowRecognizer(
        model=model,
        channel_mean=channel_mean,
        channel_std=channel_std,
        label_classes=label_classes,
        target_len=TARGET_LEN,
        window_size=WINDOW_SIZE,
        hop_size=HOP_SIZE,
        conf_threshold=CONF_THRESHOLD,
        stable_votes_required=STABLE_VOTES_REQUIRED,
        repeat_suppression=REPEAT_SUPPRESSION,
        print_topk=PRINT_TOPK,
        print_every_window=PRINT_EVERY_WINDOW,
    )

    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.5)
    except serial.SerialException as e:
        print(f"Could not open serial port: {e}")
        return

    time.sleep(2.0)
    ser.reset_input_buffer()

    print("Listening with fixed-window inference... Ctrl+C to stop.")
    print(f"WINDOW_SIZE={WINDOW_SIZE}, OVERLAP={WINDOW_OVERLAP}, HOP_SIZE={HOP_SIZE}")

    last_count = None
    drop_count = 0

    try:
        while True:
            pkt = read_imu_sample(ser)
            if pkt is None:
                continue

            count, sample6 = pkt

            if last_count is not None:
                expected = (last_count + 1) & 0xFFFF
                if count != expected:
                    delta = (count - expected) & 0xFFFF
                    drop_count += delta
                    print(f"[warn] dropped packets total={drop_count}")

            last_count = count

            result = recognizer.process_sample(sample6)
            if result is not None:
                pred_label, conf, topk = result
                print(f"{pred_label}    conf={conf:.3f}    topk={topk}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()