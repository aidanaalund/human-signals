import sys
import time
import struct
import threading
from collections import deque
from pathlib import Path
import csv

import serial
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
import pyqtgraph as pg


PORT = "COM5"          # change this
BAUD = 230400

PACKET_FMT = "<BBHIhhhhhh"
PACKET_SIZE = struct.calcsize(PACKET_FMT)

WINDOW_SEC = 5
DISPLAY_HZ = 50
MAX_SAMPLES = 1200
REPORT_INTERVAL = 2.0  # seconds

DATASET_ROOT = Path("dataset1")


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

        # Recording state
        self.recording_active = False
        self.recording_label = None
        self.recording_key = None   # Qt key code
        self.recording_start_host = None
        self.recording_start_mcu = None
        self.capture_samples = []   # list of dicts


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
    if label.isalnum():
        return label
    raise ValueError(f"Unsupported label: {label!r}")


def next_file_path(label: str) -> Path:
    label = sanitize_label(label)
    out_dir = DATASET_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    existing = sorted(out_dir.glob(f"{label}_{ts}_*.csv"))
    idx = len(existing) + 1
    return out_dir / f"{label}_{ts}_{idx:03d}.csv"


def save_capture_to_csv(label: str, samples):
    if not samples:
        print(f"[WARN] No samples captured for label {label!r}; nothing saved.")
        return None

    path = next_file_path(label)

    t0_mcu = samples[0]["t_us_mcu"]
    t0_host = samples[0]["t_host_s"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label",
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
                label,
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

    print(f"[SAVED] {label!r}: {len(samples)} samples -> {path}")
    return path


def serial_worker(buffers: SharedBuffers, stop_event: threading.Event):
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.5)
    except serial.SerialException as e:
        print(f"Could not open serial port: {e}")
        return

    time.sleep(2.0)  # allow board reset
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
                    print(
                        f"Sample rate: {rate:.2f} Hz | "
                        f"Last count: {count} | Dropped: {buffers.drop_count}"
                    )
                    buffers.report_t0 = now
                    buffers.report_count0 = count

    ser.close()


class IMUWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU Binary Plotter + Dataset Recorder")
        self.resize(1200, 800)

        DATASET_ROOT.mkdir(parents=True, exist_ok=True)

        self.buffers = SharedBuffers()
        self.stop_event = threading.Event()

        self.worker = threading.Thread(
            target=serial_worker,
            args=(self.buffers, self.stop_event),
            daemon=True
        )
        self.worker.start()

        self.central = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.central)

        # Accelerometer plot
        self.p_acc = self.central.addPlot(row=0, col=0, title="Accelerometer")
        self.p_acc.showGrid(x=True, y=True)
        self.p_acc.setLabel("left", "g")
        self.p_acc.setLabel("bottom", "Time", "s")
        self.p_acc.setYRange(-2.5, 2.5)

        self.acc_x_curve = self.p_acc.plot(pen='r')
        self.acc_y_curve = self.p_acc.plot(pen='g')
        self.acc_z_curve = self.p_acc.plot(pen='b')

        # Gyroscope plot
        self.p_gyro = self.central.addPlot(row=1, col=0, title="Gyroscope")
        self.p_gyro.showGrid(x=True, y=True)
        self.p_gyro.setLabel("left", "deg/s")
        self.p_gyro.setLabel("bottom", "Time", "s")
        self.p_gyro.setYRange(-300, 300)

        self.gyro_x_curve = self.p_gyro.plot(pen='r')
        self.gyro_y_curve = self.p_gyro.plot(pen='g')
        self.gyro_z_curve = self.p_gyro.plot(pen='b')

        self.status = QtWidgets.QLabel("Starting...")
        self.statusBar().addWidget(self.status)

        self.gui_last_time = time.perf_counter()
        self.gui_fps = 0.0

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(int(1000 / DISPLAY_HZ))

        self.setFocusPolicy(Qt.StrongFocus)

    def start_recording(self, label: str, key_code: int):
        with self.buffers.lock:
            if self.buffers.recording_active:
                print("[WARN] Already recording; ignoring new key press.")
                return

            self.buffers.recording_active = True
            self.buffers.recording_label = label
            self.buffers.recording_key = key_code
            self.buffers.recording_start_host = time.perf_counter() - self.buffers.start_time
            self.buffers.recording_start_mcu = None
            self.buffers.capture_samples = []

        print(f"[REC START] label={label!r}")

    def stop_recording(self):
        with self.buffers.lock:
            if not self.buffers.recording_active:
                return

            label = self.buffers.recording_label
            samples = list(self.buffers.capture_samples)

            self.buffers.recording_active = False
            self.buffers.recording_label = None
            self.buffers.recording_key = None
            self.buffers.recording_start_host = None
            self.buffers.recording_start_mcu = None
            self.buffers.capture_samples = []

        save_capture_to_csv(label, samples)

    def discard_recording(self):
        with self.buffers.lock:
            if not self.buffers.recording_active:
                return
            label = self.buffers.recording_label
            n = len(self.buffers.capture_samples)

            self.buffers.recording_active = False
            self.buffers.recording_label = None
            self.buffers.recording_key = None
            self.buffers.recording_start_host = None
            self.buffers.recording_start_mcu = None
            self.buffers.capture_samples = []

        print(f"[DISCARDED] label={label!r}, samples={n}")

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            event.ignore()
            return

        key = event.key()

        if key == Qt.Key_Escape:
            self.discard_recording()
            event.accept()
            return

        text = event.text()

        # Accept letters and digits only
        if text and text.isalnum() and len(text) == 1:
            # event.text() already reflects shift state:
            # 'a' vs 'A'
            self.start_recording(text, key)
            event.accept()
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            event.ignore()
            return

        with self.buffers.lock:
            active = self.buffers.recording_active
            active_key = self.buffers.recording_key

        if active and event.key() == active_key:
            self.stop_recording()
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
            recording_label = self.buffers.recording_label
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
            rec_text = f"REC '{recording_label}' ({capture_len} samples)"

        self.status.setText(
            f"{rec_text} | Last count: {last_count} | Dropped: {drop_count} | "
            f"Measured rate: {measured_rate:.2f} Hz | "
            f"Window rate: {window_rate:.2f} Hz | "
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