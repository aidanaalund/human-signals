import time
import struct
from collections import deque
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
# RUN_DIR = Path("model_runs") / "78_acc_cnn_run_20260312_003648"   # change this
RUN_DIR = Path("model_runs") / "90_acc_cnnbilstm_run_20260312_014301"   # change this
MODEL_PATH = RUN_DIR / "imu_char_cnn.keras"
MEAN_PATH = RUN_DIR / "channel_mean.npy"
STD_PATH = RUN_DIR / "channel_std.npy"
CLASSES_PATH = RUN_DIR / "label_classes.npy"

TARGET_LEN = 64


# =========================
# Segmentation / inference tuning
# =========================
START_THRESHOLD = 1.00
STOP_THRESHOLD = 0.45
QUIET_SAMPLES_TO_STOP = 5
MIN_ACTIVE_SAMPLES = 18
MAX_ACTIVE_SAMPLES = 70
CONF_THRESHOLD = 0.60
PRE_ROLL = 8
COOLDOWN_SAMPLES = 2

PRINT_SCORES = False

WINDOW_SIZE = 80
WINDOW_OVERLAP = 60

# safety check
if WINDOW_OVERLAP >= WINDOW_SIZE:
    raise ValueError("WINDOW_OVERLAP must be smaller than WINDOW_SIZE")


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

        # accel motion relative to gravity + scaled gyro motion
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


# from collections import deque
# import numpy as np


# class RealTimeCharRecognizer:
#     def __init__(
#         self,
#         model,
#         channel_mean,
#         channel_std,
#         label_classes,
#         target_len=64,
#         pre_roll=6,
#         start_threshold=1.2,
#         stop_threshold=0.35,
#         quiet_samples_to_stop=6,
#         min_active_samples=18,
#         max_active_samples=100,
#         conf_threshold=0.65,
#         cooldown_samples=8,
#         window_size=48,
#         window_overlap=32,
#         use_full_segment_vote=True,
#     ):
#         self.model = model
#         self.channel_mean = channel_mean.reshape(1, 1, -1).astype(np.float32)
#         self.channel_std = channel_std.reshape(1, 1, -1).astype(np.float32)
#         self.label_classes = np.array(label_classes)

#         self.target_len = target_len
#         self.pre_roll = pre_roll
#         self.start_threshold = start_threshold
#         self.stop_threshold = stop_threshold
#         self.quiet_samples_to_stop = quiet_samples_to_stop
#         self.min_active_samples = min_active_samples
#         self.max_active_samples = max_active_samples
#         self.conf_threshold = conf_threshold
#         self.cooldown_samples = cooldown_samples

#         self.window_size = window_size
#         self.window_overlap = window_overlap
#         if self.window_overlap >= self.window_size:
#             raise ValueError("window_overlap must be smaller than window_size")
#         self.hop_size = self.window_size - self.window_overlap

#         self.use_full_segment_vote = use_full_segment_vote

#         self.pre_buffer = deque(maxlen=pre_roll)
#         self.active_segment = []
#         self.is_writing = False
#         self.quiet_count = 0
#         self.cooldown_count = 0

#         # hybrid-window state
#         self.window_probs = []
#         self.next_window_start = 0

#     def motion_score(self, sample6):
#         ax, ay, az, gx, gy, gz = sample6
#         acc_mag = np.sqrt(ax * ax + ay * ay + az * az)
#         gyro_mag = np.sqrt(gx * gx + gy * gy + gz * gz)

#         acc_motion = abs(acc_mag - 1.0)
#         score = 3.0 * acc_motion + 0.03 * gyro_mag
#         return float(score)

#     def prepare_for_model(self, raw_segment):
#         x = preprocess_sample(raw_segment, center=True, add_magnitude=True)
#         x = resample_sequence(x, self.target_len)
#         x = x[np.newaxis, :, :]
#         x = (x - self.channel_mean) / (self.channel_std + 1e-8)
#         return x.astype(np.float32)

#     def predict_probs(self, raw_segment):
#         x = self.prepare_for_model(raw_segment)
#         probs = self.model.predict(x, verbose=0)[0]
#         return probs.astype(np.float32)

#     def _reset_active_state(self):
#         self.active_segment = []
#         self.is_writing = False
#         self.quiet_count = 0
#         self.window_probs = []
#         self.next_window_start = 0

#     def _start_segment(self):
#         self.is_writing = True
#         self.active_segment = list(self.pre_buffer)
#         self.quiet_count = 0
#         self.window_probs = []
#         self.next_window_start = 0

#     def _collect_window_predictions(self):
#         """
#         Run overlapping-window inference on any new windows that have become available.
#         """
#         seg_len = len(self.active_segment)

#         while self.next_window_start + self.window_size <= seg_len:
#             start = self.next_window_start
#             end = start + self.window_size
#             window = np.asarray(self.active_segment[start:end], dtype=np.float32)

#             probs = self.predict_probs(window)
#             self.window_probs.append(probs)

#             self.next_window_start += self.hop_size

#     def _finalize_prediction(self):
#         segment = np.asarray(self.active_segment, dtype=np.float32)

#         if len(segment) < self.min_active_samples:
#             return None

#         prob_list = list(self.window_probs)

#         if self.use_full_segment_vote:
#             full_probs = self.predict_probs(segment)
#             prob_list.append(full_probs)

#         if not prob_list:
#             # fallback in case segment ended before enough samples for a window
#             prob_list.append(self.predict_probs(segment))

#         avg_probs = np.mean(np.stack(prob_list, axis=0), axis=0)
#         pred_idx = int(np.argmax(avg_probs))
#         pred_label = str(self.label_classes[pred_idx])
#         conf = float(avg_probs[pred_idx])

#         if conf < self.conf_threshold:
#             return None

#         return pred_label, conf, len(segment), len(prob_list)

#     def process_sample(self, sample6):
#         """
#         Returns:
#             None
#             or (pred_label, conf, seg_len, num_votes)
#         """
#         score = self.motion_score(sample6)

#         if PRINT_SCORES:
#             print(f"score={score:.3f}")

#         if self.cooldown_count > 0:
#             self.cooldown_count -= 1

#         if not self.is_writing:
#             self.pre_buffer.append(sample6)

#             if self.cooldown_count == 0 and score >= self.start_threshold:
#                 self._start_segment()

#             return None

#         # currently writing
#         self.active_segment.append(sample6)

#         # new overlapping windows, if available
#         self._collect_window_predictions()

#         if score < self.stop_threshold:
#             self.quiet_count += 1
#         else:
#             self.quiet_count = 0

#         should_stop = (
#             self.quiet_count >= self.quiet_samples_to_stop
#             or len(self.active_segment) >= self.max_active_samples
#         )

#         if not should_stop:
#             return None

#         result = self._finalize_prediction()

#         self.pre_buffer.clear()
#         self.cooldown_count = self.cooldown_samples
#         self._reset_active_state()

#         return result


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
    # recognizer = RealTimeCharRecognizer(
    #     model=model,
    #     channel_mean=channel_mean,
    #     channel_std=channel_std,
    #     label_classes=label_classes,
    #     target_len=TARGET_LEN,
    #     pre_roll=PRE_ROLL,
    #     start_threshold=START_THRESHOLD,
    #     stop_threshold=STOP_THRESHOLD,
    #     quiet_samples_to_stop=QUIET_SAMPLES_TO_STOP,
    #     min_active_samples=MIN_ACTIVE_SAMPLES,
    #     max_active_samples=MAX_ACTIVE_SAMPLES,
    #     conf_threshold=CONF_THRESHOLD,
    #     cooldown_samples=COOLDOWN_SAMPLES,
    #     window_size=WINDOW_SIZE,
    #     window_overlap=WINDOW_OVERLAP,
    #     use_full_segment_vote=True,
    # )

    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.5)
    except serial.SerialException as e:
        print(f"Could not open serial port: {e}")
        return

    time.sleep(2.0)
    ser.reset_input_buffer()

    print("Listening... write a character. Ctrl+C to stop.")

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
                pred_label, conf, seg_len = result
                print(f"{pred_label}    conf={conf:.3f}    len={seg_len}")
                
            # result = recognizer.process_sample(sample6)
            # if result is not None:
            #     pred_label, conf, seg_len, num_votes = result
            #     print(f"{pred_label}    conf={conf:.3f}    len={seg_len}    votes={num_votes}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()