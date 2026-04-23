import sys
import time
import struct
import threading
import asyncio
from collections import deque

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from bleak import BleakClient, BleakScanner


DEVICE_NAME = "MG24_IMU"
CHAR_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"

PACKET_FMT = "<BBHIhhhhhh"
PACKET_SIZE = struct.calcsize(PACKET_FMT)

WINDOW_SEC = 5
DISPLAY_HZ = 208
MAX_SAMPLES = 1200
REPORT_INTERVAL = 2.0  # seconds

G = 9.80665
COMPLEMENTARY_ALPHA = 0.98

# Stationary detection / ZUPT
ACC_NORM_STILL_THRESH_G = 0.08
GYRO_STILL_THRESH_DPS = 8.0

# Startup calibration
CALIBRATION_SAMPLES = 300  # about 1.4 s at ~208 Hz


def euler_to_rotmat(roll, pitch, yaw):
    """
    Rotation matrix from body frame to world frame.
    Uses ZYX convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr]
    ])


class SharedBuffers:
    def __init__(self, maxlen=MAX_SAMPLES):
        self.lock = threading.Lock()

        self.t = deque(maxlen=maxlen)
        self.count = deque(maxlen=maxlen)

        self.ax = deque(maxlen=maxlen)
        self.ay = deque(maxlen=maxlen)
        self.az = deque(maxlen=maxlen)

        self.gx = deque(maxlen=maxlen)
        self.gy = deque(maxlen=maxlen)
        self.gz = deque(maxlen=maxlen)

        # computed signals
        self.lax = deque(maxlen=maxlen)
        self.lay = deque(maxlen=maxlen)
        self.laz = deque(maxlen=maxlen)

        self.vx = deque(maxlen=maxlen)
        self.vy = deque(maxlen=maxlen)
        self.vz = deque(maxlen=maxlen)

        self.px = deque(maxlen=maxlen)
        self.py = deque(maxlen=maxlen)
        self.pz = deque(maxlen=maxlen)

        self.start_time = time.perf_counter()
        self.last_count = None
        self.drop_count = 0

        self.report_t0 = None
        self.report_count0 = None
        self.latest_rate_hz = 0.0

        self.connected = False
        self.status_text = "Starting..."

        # incremental integration state
        self.last_sample_t = None

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.prev_linax = 0.0
        self.prev_linay = 0.0
        self.prev_linaz = 0.0

        self.vx_curr = 0.0
        self.vy_curr = 0.0
        self.vz_curr = 0.0

        self.px_curr = 0.0
        self.py_curr = 0.0
        self.pz_curr = 0.0

        # calibration state
        self.calibrated = False
        self.cal_ax = []
        self.cal_ay = []
        self.cal_az = []
        self.cal_gx = []
        self.cal_gy = []
        self.cal_gz = []

        self.ax_bias_g = 0.0
        self.ay_bias_g = 0.0
        self.az_bias_g = 0.0

        self.gx_bias_dps = 0.0
        self.gy_bias_dps = 0.0
        self.gz_bias_dps = 0.0


class BLEIMUReceiver:
    def __init__(self, buffers: SharedBuffers, stop_event: threading.Event):
        self.buffers = buffers
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

    def _process_incremental_motion(self, t, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps):
        b = self.buffers

        # collect startup calibration samples while IMU is still
        if not b.calibrated:
            b.cal_ax.append(ax_g)
            b.cal_ay.append(ay_g)
            b.cal_az.append(az_g)

            b.cal_gx.append(gx_dps)
            b.cal_gy.append(gy_dps)
            b.cal_gz.append(gz_dps)

            # hold outputs at zero during calibration
            b.lax.append(0.0)
            b.lay.append(0.0)
            b.laz.append(0.0)

            b.vx.append(0.0)
            b.vy.append(0.0)
            b.vz.append(0.0)

            b.px.append(0.0)
            b.py.append(0.0)
            b.pz.append(0.0)

            if len(b.cal_ax) >= CALIBRATION_SAMPLES:
                b.gx_bias_dps = float(np.mean(b.cal_gx))
                b.gy_bias_dps = float(np.mean(b.cal_gy))
                b.gz_bias_dps = float(np.mean(b.cal_gz))

                b.ax_bias_g = float(np.mean(b.cal_ax))
                b.ay_bias_g = float(np.mean(b.cal_ay))

                z_mean = float(np.mean(b.cal_az))
                b.az_bias_g = z_mean - np.sign(z_mean) * 1.0 if z_mean != 0 else 0.0

                ax0 = ax_g - b.ax_bias_g
                ay0 = ay_g - b.ay_bias_g
                az0 = az_g - b.az_bias_g

                b.roll = np.arctan2(ay0, az0)
                b.pitch = np.arctan2(-ax0, np.sqrt(ay0**2 + az0**2))
                b.yaw = 0.0

                b.calibrated = True
                b.status_text = "Connected and calibrated"

            else:
                b.status_text = f"Calibrating... {len(b.cal_ax)}/{CALIBRATION_SAMPLES}"

            b.last_sample_t = t
            return

        dt = 0.0 if b.last_sample_t is None else (t - b.last_sample_t)
        b.last_sample_t = t

        if dt <= 0 or dt > 0.1:
            # skip bad dt but keep outputs continuous
            b.lax.append(b.prev_linax)
            b.lay.append(b.prev_linay)
            b.laz.append(b.prev_linaz)

            b.vx.append(b.vx_curr)
            b.vy.append(b.vy_curr)
            b.vz.append(b.vz_curr)

            b.px.append(b.px_curr)
            b.py.append(b.py_curr)
            b.pz.append(b.pz_curr)
            return

        # bias-correct sensors
        gx_corr_dps = gx_dps - b.gx_bias_dps
        gy_corr_dps = gy_dps - b.gy_bias_dps
        gz_corr_dps = gz_dps - b.gz_bias_dps

        ax_corr_g = ax_g - b.ax_bias_g
        ay_corr_g = ay_g - b.ay_bias_g
        az_corr_g = az_g - b.az_bias_g

        gx_rad = np.deg2rad(gx_corr_dps)
        gy_rad = np.deg2rad(gy_corr_dps)
        gz_rad = np.deg2rad(gz_corr_dps)

        # accel tilt estimate
        roll_acc = np.arctan2(ay_corr_g, az_corr_g)
        pitch_acc = np.arctan2(
            -ax_corr_g,
            np.sqrt(ay_corr_g**2 + az_corr_g**2)
        )

        # gyro prediction + complementary correction
        roll_gyro = b.roll + gx_rad * dt
        pitch_gyro = b.pitch + gy_rad * dt
        yaw_gyro = b.yaw + gz_rad * dt

        b.roll = COMPLEMENTARY_ALPHA * roll_gyro + (1.0 - COMPLEMENTARY_ALPHA) * roll_acc
        b.pitch = COMPLEMENTARY_ALPHA * pitch_gyro + (1.0 - COMPLEMENTARY_ALPHA) * pitch_acc
        b.yaw = yaw_gyro

        # body accel -> world accel
        a_body = np.array([ax_corr_g, ay_corr_g, az_corr_g]) * G
        R_bw = euler_to_rotmat(b.roll, b.pitch, b.yaw)
        a_world = R_bw @ a_body

        # gravity removal
        linax, linay, linaz = a_world - np.array([0.0, 0.0, G])

        # stationary detection
        acc_norm_g = np.sqrt(ax_corr_g**2 + ay_corr_g**2 + az_corr_g**2)
        gyro_norm_dps = np.sqrt(gx_corr_dps**2 + gy_corr_dps**2 + gz_corr_dps**2)
        stationary = (
            abs(acc_norm_g - 1.0) < ACC_NORM_STILL_THRESH_G
            and gyro_norm_dps < GYRO_STILL_THRESH_DPS
        )

        # trapezoidal accel -> velocity
        vx_new = b.vx_curr + 0.5 * (b.prev_linax + linax) * dt
        vy_new = b.vy_curr + 0.5 * (b.prev_linay + linay) * dt
        vz_new = b.vz_curr + 0.5 * (b.prev_linaz + linaz) * dt

        # zero velocity update
        if stationary:
            vx_new = 0.0
            vy_new = 0.0
            vz_new = 0.0

        # trapezoidal velocity -> position
        px_new = b.px_curr + 0.5 * (b.vx_curr + vx_new) * dt
        py_new = b.py_curr + 0.5 * (b.vy_curr + vy_new) * dt
        pz_new = b.pz_curr + 0.5 * (b.vz_curr + vz_new) * dt

        # store state
        b.prev_linax = linax
        b.prev_linay = linay
        b.prev_linaz = linaz

        b.vx_curr = vx_new
        b.vy_curr = vy_new
        b.vz_curr = vz_new

        b.px_curr = px_new
        b.py_curr = py_new
        b.pz_curr = pz_new

        # append outputs
        b.lax.append(linax)
        b.lay.append(linay)
        b.laz.append(linaz)

        b.vx.append(vx_new)
        b.vy.append(vy_new)
        b.vz.append(vz_new)

        b.px.append(px_new)
        b.py.append(py_new)
        b.pz.append(pz_new)

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
        t = now - self.buffers.start_time

        gx = gx_d10 / 10.0
        gy = gy_d10 / 10.0
        gz = gz_d10 / 10.0

        ax = ax_mg / 1000.0
        ay = ay_mg / 1000.0
        az = az_mg / 1000.0

        with self.buffers.lock:
            if self.buffers.last_count is not None:
                expected = (self.buffers.last_count + 1) & 0xFFFF
                if count != expected:
                    delta = (count - expected) & 0xFFFF
                    self.buffers.drop_count += delta

            self.buffers.last_count = count

            self.buffers.t.append(t)
            self.buffers.count.append(count)

            self.buffers.ax.append(ax)
            self.buffers.ay.append(ay)
            self.buffers.az.append(az)

            self.buffers.gx.append(gx)
            self.buffers.gy.append(gy)
            self.buffers.gz.append(gz)

            self._process_incremental_motion(t, ax, ay, az, gx, gy, gz)

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
                        f"Last count: {count} | Dropped: {self.buffers.drop_count} | "
                        f"Pos: ({self.buffers.px_curr:.3f}, {self.buffers.py_curr:.3f}, {self.buffers.pz_curr:.3f}) m"
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
        self.setWindowTitle("IMU BLE Plotter")
        self.resize(1200, 1000)

        self.buffers = SharedBuffers()
        self.stop_event = threading.Event()

        self.receiver = BLEIMUReceiver(self.buffers, self.stop_event)
        self.receiver.start()

        self.central = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.central)

        self.p_acc = self.central.addPlot(row=0, col=0, title="Accelerometer")
        self.p_acc.showGrid(x=True, y=True)
        self.p_acc.setLabel("left", "g")
        self.p_acc.setLabel("bottom", "Time", "s")
        self.p_acc.setYRange(-2.5, 2.5)

        self.acc_x_curve = self.p_acc.plot(pen="r", name="Ax")
        self.acc_y_curve = self.p_acc.plot(pen="g", name="Ay")
        self.acc_z_curve = self.p_acc.plot(pen="b", name="Az")

        self.p_gyro = self.central.addPlot(row=1, col=0, title="Gyroscope")
        self.p_gyro.showGrid(x=True, y=True)
        self.p_gyro.setLabel("left", "deg/s")
        self.p_gyro.setLabel("bottom", "Time", "s")
        self.p_gyro.setYRange(-300, 300)

        self.gyro_x_curve = self.p_gyro.plot(pen="r", name="Gx")
        self.gyro_y_curve = self.p_gyro.plot(pen="g", name="Gy")
        self.gyro_z_curve = self.p_gyro.plot(pen="b", name="Gz")

        self.p_linacc = self.central.addPlot(row=2, col=0, title="Linear Acceleration (Gravity Removed)")
        self.p_linacc.setYRange(-1.0, 1.0)
        self.p_linacc.showGrid(x=True, y=True)
        self.p_linacc.setLabel("left", "m/s²")
        self.p_linacc.setLabel("bottom", "Time", "s")

        self.linacc_x_curve = self.p_linacc.plot(pen="r", name="Lx")
        self.linacc_y_curve = self.p_linacc.plot(pen="g", name="Ly")
        self.linacc_z_curve = self.p_linacc.plot(pen="b", name="Lz")

        self.p_pos = self.central.addPlot(row=3, col=0, title="Position (Incremental Double Integrated)")
        self.p_pos.showGrid(x=True, y=True)
        self.p_pos.setLabel("left", "m")
        self.p_pos.setLabel("bottom", "Time", "s")
        self.p_pos.setYRange(-3.0, 3.0)
        self.p_pos.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)

        self.pos_x_curve = self.p_pos.plot(pen="r", name="Px")
        self.pos_y_curve = self.p_pos.plot(pen="g", name="Py")
        self.pos_z_curve = self.p_pos.plot(pen="b", name="Pz")

        self.status = QtWidgets.QLabel("Starting...")
        self.statusBar().addWidget(self.status)

        self.gui_last_time = time.perf_counter()
        self.gui_fps = 0.0

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(int(1000 / DISPLAY_HZ))

    def update_plots(self):
        now = time.perf_counter()
        dt_gui = now - self.gui_last_time
        self.gui_last_time = now
        if dt_gui > 0:
            self.gui_fps = 1.0 / dt_gui

        with self.buffers.lock:
            if len(self.buffers.t) < 2:
                self.status.setText(self.buffers.status_text)
                return

            t = np.array(self.buffers.t, dtype=float)
            count = np.array(self.buffers.count, dtype=np.uint16)

            ax = np.array(self.buffers.ax, dtype=float)
            ay = np.array(self.buffers.ay, dtype=float)
            az = np.array(self.buffers.az, dtype=float)

            gx = np.array(self.buffers.gx, dtype=float)
            gy = np.array(self.buffers.gy, dtype=float)
            gz = np.array(self.buffers.gz, dtype=float)

            lax = np.array(self.buffers.lax, dtype=float)
            lay = np.array(self.buffers.lay, dtype=float)
            laz = np.array(self.buffers.laz, dtype=float)

            px = np.array(self.buffers.px, dtype=float)
            py = np.array(self.buffers.py, dtype=float)
            pz = np.array(self.buffers.pz, dtype=float)

            last_count = self.buffers.last_count
            drop_count = self.buffers.drop_count
            measured_rate = self.buffers.latest_rate_hz
            connected = self.buffers.connected
            status_text = self.buffers.status_text
            calibrated = self.buffers.calibrated

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

        lax = lax[mask]
        lay = lay[mask]
        laz = laz[mask]

        px = px[mask]
        py = py[mask]
        pz = pz[mask]

        self.acc_x_curve.setData(tt, ax)
        self.acc_y_curve.setData(tt, ay)
        self.acc_z_curve.setData(tt, az)

        self.gyro_x_curve.setData(tt, gx)
        self.gyro_y_curve.setData(tt, gy)
        self.gyro_z_curve.setData(tt, gz)

        self.linacc_x_curve.setData(tt, lax)
        self.linacc_y_curve.setData(tt, lay)
        self.linacc_z_curve.setData(tt, laz)

        self.pos_x_curve.setData(tt, px)
        self.pos_y_curve.setData(tt, py)
        self.pos_z_curve.setData(tt, pz)

        self.p_acc.setXRange(t_start, t_end, padding=0)
        self.p_gyro.setXRange(t_start, t_end, padding=0)
        self.p_linacc.setXRange(t_start, t_end, padding=0)
        self.p_pos.setXRange(t_start, t_end, padding=0)

        window_rate = 0.0
        if len(tt) >= 2:
            dt = tt[-1] - tt[0]
            if dt > 0:
                dc = (int(cc[-1]) - int(cc[0])) & 0xFFFF
                window_rate = dc / dt

        conn_text = "Connected" if connected else "Disconnected"
        cal_text = "Calibrated" if calibrated else "Not calibrated"

        pos_text = ""
        if len(px) > 0:
            pos_text = f" | Pos: ({px[-1]:.3f}, {py[-1]:.3f}, {pz[-1]:.3f}) m"

        self.status.setText(
            f"{conn_text} | {cal_text} | {status_text} | "
            f"Last count: {last_count} | Dropped: {drop_count} | "
            f"Measured rate: {measured_rate:.2f} Hz | "
            f"Window rate: {window_rate:.2f} Hz | "
            f"GUI FPS: {self.gui_fps:.1f}"
            f"{pos_text}"
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