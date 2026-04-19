import sys
import time
import struct
import threading
import asyncio
from collections import deque
from pathlib import Path

import numpy as np
from tensorflow import keras
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from bleak import BleakClient, BleakScanner


# =========================
# BLE settings
# =========================
DEVICE_NAME = "MG24_IMU"
CHAR_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"

PACKET_FMT = "<BBHIhhhhhh"
PACKET_SIZE = struct.calcsize(PACKET_FMT)

# =========================
# Display settings
# =========================
WINDOW_SEC = 6
DISPLAY_HZ = 25
MAX_SAMPLES = 2000
REPORT_INTERVAL = 2.0

# =========================
# Model artifacts
# =========================
RUN_DIR = Path("BLE_model_runs") / "98_acc_128len_cnnbilstm_run_20260413_111857"
MODEL_PATH = RUN_DIR / "imu_char_cnn.keras"
MEAN_PATH = RUN_DIR / "channel_mean.npy"
STD_PATH = RUN_DIR / "channel_std.npy"
CLASSES_PATH = RUN_DIR / "label_classes.npy"

TARGET_LEN = 128

# =========================
# Segmentation / inference tuning
# =========================
START_THRESHOLD = 2.00
STOP_THRESHOLD = 2.00
QUIET_SAMPLES_TO_STOP = 10
MIN_ACTIVE_SAMPLES = 36
MAX_ACTIVE_SAMPLES = 140
CONF_THRESHOLD = 0.70
PRE_ROLL = 16
COOLDOWN_SAMPLES = 4


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

    def motion_components(self, sample6):
        """
        sample6 = [ax, ay, az, gx, gy, gz]
        accel in g, gyro in dps
        """
        ax, ay, az, gx, gy, gz = sample6

        acc_mag = np.sqrt(ax * ax + ay * ay + az * az)
        gyro_mag = np.sqrt(gx * gx + gy * gy + gz * gz)

        acc_motion = abs(acc_mag - 1.0)
        score = 3.0 * acc_motion + 0.03 * gyro_mag
        return float(acc_motion), float(gyro_mag), float(score)

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
        Returns debug dict every sample.
        """
        acc_motion, gyro_mag, score = self.motion_components(sample6)

        if self.cooldown_count > 0:
            self.cooldown_count -= 1

        if not self.is_writing:
            self.pre_buffer.append(sample6)

            event = None
            if self.cooldown_count == 0 and score >= self.start_threshold:
                self.is_writing = True
                self.active_segment = list(self.pre_buffer)
                self.quiet_count = 0
                event = "start"

            return {
                "score": score,
                "acc_motion": acc_motion,
                "gyro_mag": gyro_mag,
                "is_writing": self.is_writing,
                "event": event,
                "pred_label": None,
                "conf": None,
                "seg_len": None,
                "stop_reason": None,
            }

        self.active_segment.append(sample6)

        stop_reason = None
        if score < self.stop_threshold:
            self.quiet_count += 1
        else:
            self.quiet_count = 0

        if self.quiet_count >= self.quiet_samples_to_stop:
            stop_reason = "quiet"
        elif len(self.active_segment) >= self.max_active_samples:
            stop_reason = "max_len"

        if stop_reason is None:
            return {
                "score": score,
                "acc_motion": acc_motion,
                "gyro_mag": gyro_mag,
                "is_writing": self.is_writing,
                "event": None,
                "pred_label": None,
                "conf": None,
                "seg_len": None,
                "stop_reason": None,
            }

        segment = np.asarray(self.active_segment, dtype=np.float32)
        seg_len = len(segment)

        self.is_writing = False
        self.active_segment = []
        self.pre_buffer.clear()
        self.cooldown_count = self.cooldown_samples
        self.quiet_count = 0

        if seg_len < self.min_active_samples:
            return {
                "score": score,
                "acc_motion": acc_motion,
                "gyro_mag": gyro_mag,
                "is_writing": self.is_writing,
                "event": "reject_short",
                "pred_label": None,
                "conf": None,
                "seg_len": seg_len,
                "stop_reason": stop_reason,
            }

        pred_label, conf, _ = self.predict_segment(segment)

        if conf < self.conf_threshold:
            return {
                "score": score,
                "acc_motion": acc_motion,
                "gyro_mag": gyro_mag,
                "is_writing": self.is_writing,
                "event": "reject_conf",
                "pred_label": pred_label,
                "conf": conf,
                "seg_len": seg_len,
                "stop_reason": stop_reason,
            }

        return {
            "score": score,
            "acc_motion": acc_motion,
            "gyro_mag": gyro_mag,
            "is_writing": self.is_writing,
            "event": "accept",
            "pred_label": pred_label,
            "conf": conf,
            "seg_len": seg_len,
            "stop_reason": stop_reason,
        }


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

        self.score = deque(maxlen=maxlen)
        self.acc_motion = deque(maxlen=maxlen)
        self.gyro_mag = deque(maxlen=maxlen)
        self.writing_state = deque(maxlen=maxlen)

        # (t_host, label)
        self.seg_events = deque(maxlen=200)

        self.start_time = time.perf_counter()
        self.last_count = None
        self.drop_count = 0

        self.report_t0 = None
        self.report_count0 = None
        self.latest_rate_hz = 0.0

        self.connected = False
        self.status_text = "Starting..."
        self.latest_pred_text = ""


class BLEIMUReceiver:
    def __init__(self, buffers: SharedBuffers, recognizer, stop_event: threading.Event):
        self.buffers = buffers
        self.recognizer = recognizer
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

    def _handle_packet(self, data: bytearray):
        if len(data) != PACKET_SIZE:
            return

        try:
            h1, h2, count, t_us, gx_d10, gy_d10, gz_d10, ax_mg, ay_mg, az_mg = struct.unpack(
                PACKET_FMT, data
            )
        except struct.error:
            return

        if h1 != 0xAA or h2 != 0x55:
            return

        now = time.perf_counter()
        t_host = now - self.buffers.start_time

        gx = gx_d10 / 10.0
        gy = gy_d10 / 10.0
        gz = gz_d10 / 10.0

        ax = ax_mg / 1000.0
        ay = ay_mg / 1000.0
        az = az_mg / 1000.0

        sample6 = np.array([ax, ay, az, gx, gy, gz], dtype=np.float32)
        debug = self.recognizer.process_sample(sample6)

        with self.buffers.lock:
            if self.buffers.last_count is not None:
                expected = (self.buffers.last_count + 1) & 0xFFFF
                if count != expected:
                    delta = (count - expected) & 0xFFFF
                    self.buffers.drop_count += delta

            self.buffers.last_count = count

            self.buffers.t_host.append(t_host)
            self.buffers.t_us_mcu.append(t_us)
            self.buffers.count.append(count)

            self.buffers.ax.append(ax)
            self.buffers.ay.append(ay)
            self.buffers.az.append(az)

            self.buffers.gx.append(gx)
            self.buffers.gy.append(gy)
            self.buffers.gz.append(gz)

            self.buffers.score.append(debug["score"])
            self.buffers.acc_motion.append(debug["acc_motion"])
            self.buffers.gyro_mag.append(debug["gyro_mag"])
            self.buffers.writing_state.append(1.0 if debug["is_writing"] else 0.0)

            if debug["event"] is not None:
                if debug["event"] == "start":
                    label = "start"
                elif debug["event"] == "accept":
                    label = f"accept:{debug['pred_label']} ({debug['conf']:.2f}) len={debug['seg_len']} {debug['stop_reason']}"
                    self.buffers.latest_pred_text = label
                    print("\n", label, "\n")
                elif debug["event"] == "reject_conf":
                    label = f"reject_conf:{debug['pred_label']} ({debug['conf']:.2f}) len={debug['seg_len']} {debug['stop_reason']}"
                    self.buffers.latest_pred_text = label
                    print(label)
                elif debug["event"] == "reject_short":
                    label = f"reject_short len={debug['seg_len']} {debug['stop_reason']}"
                    self.buffers.latest_pred_text = label
                    print(label)
                else:
                    label = debug["event"]

                self.buffers.seg_events.append((t_host, label))

            if self.buffers.report_t0 is None:
                self.buffers.report_t0 = now
                self.buffers.report_count0 = count
            else:
                dt = now - self.buffers.report_t0
                if dt >= REPORT_INTERVAL:
                    dc = (count - self.buffers.report_count0) & 0xFFFF
                    rate = dc / dt if dt > 0 else 0.0
                    self.buffers.latest_rate_hz = rate
                    print(
                        f"Sample rate: {rate:.2f} Hz | "
                        f"Last count: {count} | Dropped: {self.buffers.drop_count}"
                    )
                    self.buffers.report_t0 = now
                    self.buffers.report_count0 = count

    def _notification_handler(self, sender, data: bytearray):
        self._handle_packet(data)

    async def _ble_main(self):
        while not self.stop_event.is_set():
            try:
                with self.buffers.lock:
                    self.buffers.status_text = "Scanning for BLE device..."
                    self.buffers.connected = False

                device = await self._find_device()

                if device is None:
                    with self.buffers.lock:
                        self.buffers.status_text = "Device not found; retrying..."
                    await asyncio.sleep(2.0)
                    continue

                with self.buffers.lock:
                    self.buffers.status_text = f"Connecting to {device.name}..."

                async with BleakClient(device) as client:
                    with self.buffers.lock:
                        self.buffers.connected = True
                        self.buffers.status_text = f"Connected to {device.name}"

                    await client.start_notify(CHAR_UUID, self._notification_handler)

                    while client.is_connected and not self.stop_event.is_set():
                        await asyncio.sleep(0.2)

                    try:
                        await client.stop_notify(CHAR_UUID)
                    except Exception:
                        pass

            except Exception as e:
                with self.buffers.lock:
                    self.buffers.status_text = f"BLE error: {e}"
                print(f"BLE error: {e}")

            with self.buffers.lock:
                self.buffers.connected = False

            if not self.stop_event.is_set():
                await asyncio.sleep(2.0)


class IMUWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU Threshold Segmentation Debugger")
        self.resize(1400, 980)

        self.model = keras.models.load_model(MODEL_PATH)
        self.channel_mean = np.load(MEAN_PATH)
        self.channel_std = np.load(STD_PATH)
        self.label_classes = np.load(CLASSES_PATH, allow_pickle=True)

        self.recognizer = RealTimeCharRecognizer(
            model=self.model,
            channel_mean=self.channel_mean,
            channel_std=self.channel_std,
            label_classes=self.label_classes,
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

        self.buffers = SharedBuffers()
        self.stop_event = threading.Event()

        self.receiver = BLEIMUReceiver(self.buffers, self.recognizer, self.stop_event)
        self.receiver.start()

        self.seg_event_lines = []
        self.seg_event_texts = []

        self._build_ui()

        self.gui_last_time = time.perf_counter()
        self.gui_fps = 0.0

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(int(1000 / DISPLAY_HZ))

    def _build_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        title = QtWidgets.QLabel("Live Threshold Segmentation Debug")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: bold;")
        layout.addWidget(title)

        info = QtWidgets.QLabel(
            f"start={START_THRESHOLD:.2f} | stop={STOP_THRESHOLD:.2f} | "
            f"quiet={QUIET_SAMPLES_TO_STOP} | min_len={MIN_ACTIVE_SAMPLES} | "
            f"max_len={MAX_ACTIVE_SAMPLES} | conf={CONF_THRESHOLD:.2f}"
        )
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("font-size: 14px;")
        layout.addWidget(info)

        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics, stretch=1)

        # Accelerometer plot
        self.p_acc = self.graphics.addPlot(row=0, col=0, title="Accelerometer")
        self.p_acc.showGrid(x=True, y=True)
        self.p_acc.setLabel("left", "g")                    
        self.p_acc.setLabel("bottom", "Time", "s")
        self.p_acc.setYRange(-2.5, 2.5)
        self.acc_x_curve = self.p_acc.plot(pen='r')
        self.acc_y_curve = self.p_acc.plot(pen='g')
        self.acc_z_curve = self.p_acc.plot(pen='b')

        # Gyro plot
        self.p_gyro = self.graphics.addPlot(row=1, col=0, title="Gyroscope")
        self.p_gyro.showGrid(x=True, y=True)
        self.p_gyro.setLabel("left", "deg/s")
        self.p_gyro.setLabel("bottom", "Time", "s")
        self.p_gyro.setYRange(-300, 300)
        self.gyro_x_curve = self.p_gyro.plot(pen='r')
        self.gyro_y_curve = self.p_gyro.plot(pen='g')
        self.gyro_z_curve = self.p_gyro.plot(pen='b')

        # Segmentation debug plot
        self.p_seg = self.graphics.addPlot(row=2, col=0, title="Segmentation Debug")
        self.p_seg.showGrid(x=True, y=True)
        self.p_seg.setLabel("left", "Score / State")
        self.p_seg.setLabel("bottom", "Time", "s")

        self.score_curve = self.p_seg.plot(pen='y')
        self.acc_motion_curve = self.p_seg.plot(pen='c')
        self.gyro_mag_curve = self.p_seg.plot(pen='m')
        self.state_curve = self.p_seg.plot(pen=pg.mkPen('w', width=2))

        # Threshold lines
        self.start_thr_line = pg.InfiniteLine(
            pos=START_THRESHOLD, angle=0, pen=pg.mkPen('g', style=QtCore.Qt.DashLine)
        )
        self.stop_thr_line = pg.InfiniteLine(
            pos=STOP_THRESHOLD, angle=0, pen=pg.mkPen('r', style=QtCore.Qt.DashLine)
        )
        self.p_seg.addItem(self.start_thr_line)
        self.p_seg.addItem(self.stop_thr_line)

        # Status bar
        self.status = QtWidgets.QLabel("Starting...")
        self.statusBar().addWidget(self.status)

    # =========================
    # PLOT UPDATE LOOP
    # =========================
    def update_plots(self):
        now = time.perf_counter()
        dt_gui = now - self.gui_last_time
        self.gui_last_time = now
        if dt_gui > 0:
            self.gui_fps = 1.0 / dt_gui

        with self.buffers.lock:
            if len(self.buffers.t_host) < 2:
                self.status.setText(self.buffers.status_text)
                return

            t = np.array(self.buffers.t_host, dtype=float)

            ax = np.array(self.buffers.ax, dtype=float)
            ay = np.array(self.buffers.ay, dtype=float)
            az = np.array(self.buffers.az, dtype=float)

            gx = np.array(self.buffers.gx, dtype=float)
            gy = np.array(self.buffers.gy, dtype=float)
            gz = np.array(self.buffers.gz, dtype=float)

            score = np.array(self.buffers.score, dtype=float)
            acc_motion = np.array(self.buffers.acc_motion, dtype=float)
            gyro_mag = np.array(self.buffers.gyro_mag, dtype=float)
            writing_state = np.array(self.buffers.writing_state, dtype=float)

            seg_events = list(self.buffers.seg_events)

            last_count = self.buffers.last_count
            drop_count = self.buffers.drop_count
            measured_rate = self.buffers.latest_rate_hz
            latest_pred_text = self.buffers.latest_pred_text
            connected = self.buffers.connected
            status_text = self.buffers.status_text

        # windowing
        t_end = t[-1]
        t_start = max(0.0, t_end - WINDOW_SEC)
        mask = t >= t_start

        tt = t[mask]

        ax = ax[mask]
        ay = ay[mask]
        az = az[mask]

        gx = gx[mask]
        gy = gy[mask]
        gz = gz[mask]

        score = score[mask]
        acc_motion = acc_motion[mask]
        gyro_mag = gyro_mag[mask]
        writing_state = writing_state[mask]

        # update curves
        self.acc_x_curve.setData(tt, ax)
        self.acc_y_curve.setData(tt, ay)
        self.acc_z_curve.setData(tt, az)

        self.gyro_x_curve.setData(tt, gx)
        self.gyro_y_curve.setData(tt, gy)
        self.gyro_z_curve.setData(tt, gz)

        self.score_curve.setData(tt, score)
        self.acc_motion_curve.setData(tt, acc_motion)
        self.gyro_mag_curve.setData(tt, gyro_mag * 0.03)  # scaled
        self.state_curve.setData(tt, writing_state * START_THRESHOLD)

        # sync x ranges
        self.p_acc.setXRange(t_start, t_end, padding=0)
        self.p_gyro.setXRange(t_start, t_end, padding=0)
        self.p_seg.setXRange(t_start, t_end, padding=0)

        # clear old markers
        for line in self.seg_event_lines:
            self.p_seg.removeItem(line)
        self.seg_event_lines.clear()

        for txt in self.seg_event_texts:
            self.p_seg.removeItem(txt)
        self.seg_event_texts.clear()

        # draw new markers
        for t_evt, label in seg_events:
            if t_evt < t_start:
                continue

            line = pg.InfiniteLine(pos=t_evt, angle=90, pen=pg.mkPen('w'))
            self.p_seg.addItem(line)
            self.seg_event_lines.append(line)

            text = pg.TextItem(label, anchor=(0, 1))
            text.setPos(t_evt, START_THRESHOLD * 1.05)
            self.p_seg.addItem(text)
            self.seg_event_texts.append(text)

        # compute window rate
        window_rate = 0.0
        if len(tt) >= 2:
            dt = tt[-1] - tt[0]
            if dt > 0:
                window_rate = len(tt) / dt

        conn_text = "Connected" if connected else "Disconnected"

        self.status.setText(
            f"{conn_text} | {status_text} | "
            f"Last count: {last_count} | Dropped: {drop_count} | "
            f"Measured: {measured_rate:.2f} Hz | Window: {window_rate:.2f} Hz | "
            f"GUI FPS: {self.gui_fps:.1f} | Pred: {latest_pred_text}"
        )

    def closeEvent(self, event):
        self.stop_event.set()
        self.receiver.join(timeout=1.0)
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = IMUWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()