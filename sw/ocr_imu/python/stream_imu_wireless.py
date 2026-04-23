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
DISPLAY_HZ = 52
MAX_SAMPLES = 1200
REPORT_INTERVAL = 2.0  # seconds


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

        self.start_time = time.perf_counter()
        self.last_count = None
        self.drop_count = 0

        self.report_t0 = None
        self.report_count0 = None
        self.latest_rate_hz = 0.0

        self.connected = False
        self.status_text = "Starting..."


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
        self.setWindowTitle("IMU BLE Plotter")
        self.resize(1200, 800)

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

            last_count = self.buffers.last_count
            drop_count = self.buffers.drop_count
            measured_rate = self.buffers.latest_rate_hz
            connected = self.buffers.connected
            status_text = self.buffers.status_text

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

        conn_text = "Connected" if connected else "Disconnected"

        self.status.setText(
            f"{conn_text} | {status_text} | "
            f"Last count: {last_count} | Dropped: {drop_count} | "
            f"Measured rate: {measured_rate:.2f} Hz | "
            f"Window rate: {window_rate:.2f} Hz | "
            f"GUI FPS: {self.gui_fps:.1f}"
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