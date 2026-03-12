import sys
import time
import struct
import threading
from collections import deque
from pathlib import Path
import csv
import random

import serial
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
import pyqtgraph as pg


PORT = "COM5"
BAUD = 230400

PACKET_FMT = "<BBHIhhhhhh"
PACKET_SIZE = struct.calcsize(PACKET_FMT)

WINDOW_SEC = 5
DISPLAY_HZ = 50
MAX_SAMPLES = 1200
REPORT_INTERVAL = 2.0

# =========================
# Dataset configuration
# =========================
DATASET_ROOT = Path("prompted_dataset_2")
SESSION_NAME = "session1"

INCLUDE_LOWERCASE = True
INCLUDE_UPPERCASE = True
INCLUDE_DIGITS = True

SAMPLES_PER_CLASS = 20

# Set to e.g. ["normal"] for just one round
SPEED_CONDITIONS = ["normal", "slow", "fast"]

SHUFFLE_PROMPTS = True
RANDOM_SEED = 42


def build_class_list():
    classes = []
    if INCLUDE_LOWERCASE:
        classes += [chr(ord('a') + i) for i in range(26)]
    if INCLUDE_UPPERCASE:
        classes += [chr(ord('A') + i) for i in range(26)]
    if INCLUDE_DIGITS:
        classes += [str(i) for i in range(10)]
    return classes


def build_prompt_list():
    rng = random.Random(RANDOM_SEED)
    classes = build_class_list()
    prompts = []

    for speed in SPEED_CONDITIONS:
        block = []
        for label in classes:
            for _ in range(SAMPLES_PER_CLASS):
                block.append({"label": label, "speed": speed})
        if SHUFFLE_PROMPTS:
            rng.shuffle(block)
        prompts.extend(block)

    return prompts


class SharedBuffers:
    def __init__(self, maxlen=MAX_SAMPLES):
        self.lock = threading.Lock()

        self.t_host = deque(maxlen=maxlen)
        self.t_us_mcu = deque(maxlen=maxlen)
        self.count = deque(maxlen=maxlen)

        self.ax = deque(maxlen=maxlen)
        self.ay = deque(maxlen=maxlen)
        self.az = deque(maxlen=maxlen)

        self.gx = deque(maxlen=maxlen)
        self.gy = deque(maxlen=maxlen)
        self.gz = deque(maxlen=maxlen)

        self.start_time = time.perf_counter()
        self.last_count = None
        self.drop_count = 0

        self.report_t0 = None
        self.report_count0 = None
        self.latest_rate_hz = 0.0

        self.recording_active = False
        self.recording_start_drop_count = 0
        self.capture_samples = []


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


def sanitize_label(label: str) -> str:
    if label.isalnum() and len(label) == 1:
        return label
    raise ValueError(f"Unsupported label: {label!r}")


def make_session_dir():
    session_dir = DATASET_ROOT / SESSION_NAME
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def next_file_path(session_dir: Path, label: str) -> Path:
    label = sanitize_label(label)
    out_dir = session_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    idx = 1
    while True:
        path = out_dir / f"{label}_{ts}_{idx:04d}.csv"
        if not path.exists():
            return path
        idx += 1


def manifest_path(session_dir: Path) -> Path:
    return session_dir / "manifest.csv"


def ensure_manifest_exists(session_dir: Path):
    mpath = manifest_path(session_dir)
    if not mpath.exists():
        with open(mpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "session",
                "label",
                "speed",
                "prompt_index",
                "file_path",
                "num_samples",
                "start_count",
                "end_count",
                "drops_during_capture",
                "measured_rate_hz",
            ])


def append_manifest_row(session_dir: Path, row):
    with open(manifest_path(session_dir), "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def save_capture_to_csv(path: Path, label: str, speed: str, samples):
    if not samples:
        print(f"[WARN] No samples captured for label={label!r}, speed={speed!r}")
        return False

    t0_mcu = samples[0]["t_us_mcu"]
    t0_host = samples[0]["t_host_s"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["meta_label", label])
        writer.writerow(["meta_speed", speed])
        writer.writerow([])
        writer.writerow([
            "count",
            "t_us_mcu",
            "t_rel_us_mcu",
            "t_host_s",
            "t_rel_host_s",
            "ax_g",
            "ay_g",
            "az_g",
            "gx_dps",
            "gy_dps",
            "gz_dps",
        ])

        for s in samples:
            writer.writerow([
                s["count"],
                s["t_us_mcu"],
                s["t_us_mcu"] - t0_mcu,
                f"{s['t_host_s']:.9f}",
                f"{s['t_host_s'] - t0_host:.9f}",
                f"{s['ax']:.6f}",
                f"{s['ay']:.6f}",
                f"{s['az']:.6f}",
                f"{s['gx']:.6f}",
                f"{s['gy']:.6f}",
                f"{s['gz']:.6f}",
            ])

    return True


def serial_worker(buffers: SharedBuffers, stop_event: threading.Event):
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.5)
    except serial.SerialException as e:
        print(f"Could not open serial port: {e}")
        return

    time.sleep(2.0)
    ser.reset_input_buffer()

    while not stop_event.is_set():
        ok = sync_header(ser)
        if not ok:
            continue

        payload = read_exactly(ser, PACKET_SIZE - 2)
        if payload is None:
            continue

        try:
            _, _, count, t_us, gx_d10, gy_d10, gz_d10, ax_mg, ay_mg, az_mg = \
                struct.unpack(PACKET_FMT, b"\xAA\x55" + payload)
        except struct.error:
            continue

        now = time.perf_counter()
        t_host = now - buffers.start_time

        gx = gx_d10 / 10.0
        gy = gy_d10 / 10.0
        gz = gz_d10 / 10.0

        ax = ax_mg / 1000.0
        ay = ay_mg / 1000.0
        az = az_mg / 1000.0

        with buffers.lock:
            if buffers.last_count is not None:
                expected = (buffers.last_count + 1) & 0xFFFF
                if count != expected:
                    delta = (count - expected) & 0xFFFF
                    buffers.drop_count += delta

            buffers.last_count = count

            buffers.t_host.append(t_host)
            buffers.t_us_mcu.append(t_us)
            buffers.count.append(count)

            buffers.ax.append(ax)
            buffers.ay.append(ay)
            buffers.az.append(az)

            buffers.gx.append(gx)
            buffers.gy.append(gy)
            buffers.gz.append(gz)

            if buffers.recording_active:
                buffers.capture_samples.append({
                    "count": count,
                    "t_us_mcu": t_us,
                    "t_host_s": t_host,
                    "ax": ax,
                    "ay": ay,
                    "az": az,
                    "gx": gx,
                    "gy": gy,
                    "gz": gz,
                })

            if buffers.report_t0 is None:
                buffers.report_t0 = now
                buffers.report_count0 = count
            else:
                dt = now - buffers.report_t0
                if dt >= REPORT_INTERVAL:
                    dc = (count - buffers.report_count0) & 0xFFFF
                    rate = dc / dt if dt > 0 else 0.0
                    buffers.latest_rate_hz = rate
                    print(f"Sample rate: {rate:.2f} Hz | Last count: {count} | Dropped: {buffers.drop_count}")
                    buffers.report_t0 = now
                    buffers.report_count0 = count

    ser.close()


class IMUWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU Prompted Dataset Collector")
        self.resize(1300, 900)

        self.session_dir = make_session_dir()
        ensure_manifest_exists(self.session_dir)

        self.prompts = build_prompt_list()
        self.prompt_index = 0

        self.buffers = SharedBuffers()
        self.stop_event = threading.Event()

        self.worker = threading.Thread(
            target=serial_worker,
            args=(self.buffers, self.stop_event),
            daemon=True
        )
        self.worker.start()

        self._build_ui()

        self.gui_last_time = time.perf_counter()
        self.gui_fps = 0.0

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(int(1000 / DISPLAY_HZ))

        self.setFocusPolicy(Qt.StrongFocus)

    def _build_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        self.prompt_label = QtWidgets.QLabel()
        self.prompt_label.setAlignment(Qt.AlignCenter)
        self.prompt_label.setStyleSheet("font-size: 36px; font-weight: bold;")

        self.progress_label = QtWidgets.QLabel()
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 18px;")

        self.help_label = QtWidgets.QLabel(
            "Hold SPACE to record prompted character | Release SPACE to save | ESC discard | BACKSPACE previous prompt"
        )
        self.help_label.setAlignment(Qt.AlignCenter)
        self.help_label.setStyleSheet("font-size: 14px;")

        layout.addWidget(self.prompt_label)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.help_label)

        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics, stretch=1)

        self.p_acc = self.graphics.addPlot(row=0, col=0, title="Accelerometer")
        self.p_acc.showGrid(x=True, y=True)
        self.p_acc.setLabel("left", "g")
        self.p_acc.setLabel("bottom", "Time", "s")
        self.p_acc.setYRange(-2.5, 2.5)
        self.acc_x_curve = self.p_acc.plot(pen='r')
        self.acc_y_curve = self.p_acc.plot(pen='g')
        self.acc_z_curve = self.p_acc.plot(pen='b')

        self.p_gyro = self.graphics.addPlot(row=1, col=0, title="Gyroscope")
        self.p_gyro.showGrid(x=True, y=True)
        self.p_gyro.setLabel("left", "deg/s")
        self.p_gyro.setLabel("bottom", "Time", "s")
        self.p_gyro.setYRange(-300, 300)
        self.gyro_x_curve = self.p_gyro.plot(pen='r')
        self.gyro_y_curve = self.p_gyro.plot(pen='g')
        self.gyro_z_curve = self.p_gyro.plot(pen='b')

        self.status = QtWidgets.QLabel("Starting...")
        self.statusBar().addWidget(self.status)

        self.refresh_prompt_display()

    def current_prompt(self):
        if 0 <= self.prompt_index < len(self.prompts):
            return self.prompts[self.prompt_index]
        return None

    def refresh_prompt_display(self):
        prompt = self.current_prompt()
        if prompt is None:
            self.prompt_label.setText("DONE")
            self.progress_label.setText("All prompts completed.")
            return

        label = prompt["label"]
        speed = prompt["speed"]

        class_counts_done = sum(
            1 for i in range(self.prompt_index)
            if self.prompts[i]["label"] == label and self.prompts[i]["speed"] == speed
        )
        class_counts_total = sum(
            1 for p in self.prompts
            if p["label"] == label and p["speed"] == speed
        )

        self.prompt_label.setText(f"Write: {label}")
        self.progress_label.setText(
            f"Speed: {speed} | Prompt {self.prompt_index + 1}/{len(self.prompts)} | "
            f"This class-speed: {class_counts_done + 1}/{class_counts_total}"
        )

    def start_recording(self):
        prompt = self.current_prompt()
        if prompt is None:
            return

        with self.buffers.lock:
            if self.buffers.recording_active:
                return
            self.buffers.recording_active = True
            self.buffers.recording_start_drop_count = self.buffers.drop_count
            self.buffers.capture_samples = []

        print(f"[REC START] label={prompt['label']!r}, speed={prompt['speed']!r}")

    def finish_recording_and_save(self):
        prompt = self.current_prompt()
        if prompt is None:
            return

        with self.buffers.lock:
            if not self.buffers.recording_active:
                return

            samples = list(self.buffers.capture_samples)
            drops_during_capture = self.buffers.drop_count - self.buffers.recording_start_drop_count
            measured_rate = self.buffers.latest_rate_hz
            self.buffers.recording_active = False
            self.buffers.capture_samples = []

        label = prompt["label"]
        speed = prompt["speed"]
        path = next_file_path(self.session_dir, label)

        ok = save_capture_to_csv(path, label, speed, samples)
        if not ok:
            return

        start_count = samples[0]["count"] if samples else ""
        end_count = samples[-1]["count"] if samples else ""

        append_manifest_row(self.session_dir, [
            time.strftime("%Y-%m-%d %H:%M:%S"),
            SESSION_NAME,
            label,
            speed,
            self.prompt_index,
            str(path.relative_to(self.session_dir)),
            len(samples),
            start_count,
            end_count,
            drops_during_capture,
            f"{measured_rate:.4f}",
        ])

        print(f"[SAVED] {label!r}, speed={speed!r}, samples={len(samples)} -> {path}")
        self.prompt_index += 1
        self.refresh_prompt_display()

    def discard_recording(self):
        with self.buffers.lock:
            if not self.buffers.recording_active:
                return
            n = len(self.buffers.capture_samples)
            self.buffers.recording_active = False
            self.buffers.capture_samples = []
        print(f"[DISCARDED] {n} samples")

    def go_to_previous_prompt(self):
        with self.buffers.lock:
            if self.buffers.recording_active:
                return
        if self.prompt_index > 0:
            self.prompt_index -= 1
            self.refresh_prompt_display()
            print(f"[PROMPT] moved back to {self.prompt_index + 1}/{len(self.prompts)}")

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            event.ignore()
            return

        if event.key() == Qt.Key_Space:
            self.start_recording()
            event.accept()
            return

        if event.key() == Qt.Key_Escape:
            self.discard_recording()
            event.accept()
            return

        if event.key() == Qt.Key_Backspace:
            self.go_to_previous_prompt()
            event.accept()
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            event.ignore()
            return

        if event.key() == Qt.Key_Space:
            self.finish_recording_and_save()
            event.accept()
            return

        super().keyReleaseEvent(event)

    def update_plots(self):
        now = time.perf_counter()
        dt_gui = now - self.gui_last_time
        self.gui_last_time = now
        if dt_gui > 0:
            self.gui_fps = 1.0 / dt_gui

        with self.buffers.lock:
            if len(self.buffers.t_host) < 2:
                return

            t = np.array(self.buffers.t_host, dtype=float)
            count = np.array(self.buffers.count, dtype=np.uint16)

            ax = np.array(self.buffers.ax, dtype=float)
            ay = np.array(self.buffers.ay, dtype=float)
            az = np.array(self.buffers.az, dtype=float)

            gx = np.array(self.buffers.gx, dtype=float)
            gy = np.array(self.buffers.gy, dtype=float)
            gz = np.array(self.buffers.gz, dtype=float)

            last_count = self.buffers.last_count
            drop_count = self.buffers.drop_count
            measured_rate = self.buffers.latest_rate_hz
            recording_active = self.buffers.recording_active
            capture_len = len(self.buffers.capture_samples)

        t_end = t[-1]
        t_start = max(0.0, t_end - WINDOW_SEC)
        mask = t >= t_start

        tt = t[mask]
        cc = count[mask]

        ax = ax[mask]
        ay = ay[mask]
        az = az[mask]

        gx = gx[mask]
        gy = gy[mask]
        gz = gz[mask]

        self.acc_x_curve.setData(tt, ax)
        self.acc_y_curve.setData(tt, ay)
        self.acc_z_curve.setData(tt, az)

        self.gyro_x_curve.setData(tt, gx)
        self.gyro_y_curve.setData(tt, gy)
        self.gyro_z_curve.setData(tt, gz)

        self.p_acc.setXRange(t_start, t_end, padding=0)
        self.p_gyro.setXRange(t_start, t_end, padding=0)

        window_rate = 0.0
        if len(tt) >= 2:
            dt = tt[-1] - tt[0]
            if dt > 0:
                dc = (int(cc[-1]) - int(cc[0])) & 0xFFFF
                window_rate = dc / dt

        rec_text = "Idle"
        if recording_active:
            rec_text = f"REC ({capture_len} samples)"

        self.status.setText(
            f"{rec_text} | Prompt {min(self.prompt_index + 1, len(self.prompts))}/{len(self.prompts)} | "
            f"Last count: {last_count} | Dropped: {drop_count} | "
            f"Measured rate: {measured_rate:.2f} Hz | Window rate: {window_rate:.2f} Hz | "
            f"GUI FPS: {self.gui_fps:.1f}"
        )

    def closeEvent(self, event):
        self.stop_event.set()
        self.worker.join(timeout=1.0)
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = IMUWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()