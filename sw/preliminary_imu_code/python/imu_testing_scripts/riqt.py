import sys
import time
import struct
import threading
from collections import deque

import serial
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg


PORT = "COM5"          # change this
BAUD = 230400

PACKET_FMT = "<BBHhhhhhh"
PACKET_SIZE = struct.calcsize(PACKET_FMT)

WINDOW_SEC = 5
DISPLAY_HZ = 50
MAX_SAMPLES = 600      # enough for ~5 s at 104 Hz


class SharedBuffers:
    def __init__(self, maxlen=MAX_SAMPLES):
        self.lock = threading.Lock()
        self.count = deque(maxlen=maxlen)
        self.t = deque(maxlen=maxlen)

        self.ax = deque(maxlen=maxlen)
        self.ay = deque(maxlen=maxlen)
        self.az = deque(maxlen=maxlen)

        self.gx = deque(maxlen=maxlen)
        self.gy = deque(maxlen=maxlen)
        self.gz = deque(maxlen=maxlen)

        self.start_time = time.perf_counter()
        self.last_count = None
        self.drop_count = 0


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
            _, _, count, gx_d10, gy_d10, gz_d10, ax_mg, ay_mg, az_mg = struct.unpack(
                PACKET_FMT, b"\xAA\x55" + payload
            )
        except struct.error:
            continue

        t = time.perf_counter() - buffers.start_time

        ax = ax_mg / 1000.0
        ay = ay_mg / 1000.0
        az = az_mg / 1000.0

        gx = gx_d10 / 10.0
        gy = gy_d10 / 10.0
        gz = gz_d10 / 10.0

        with buffers.lock:
            if buffers.last_count is not None:
                expected = (buffers.last_count + 1) & 0xFFFF
                if count != expected:
                    delta = (count - expected) & 0xFFFF
                    buffers.drop_count += delta

            buffers.last_count = count

            buffers.count.append(count)
            buffers.t.append(t)

            buffers.ax.append(ax)
            buffers.ay.append(ay)
            buffers.az.append(az)

            buffers.gx.append(gx)
            buffers.gy.append(gy)
            buffers.gz.append(gz)

    ser.close()


class IMUWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU FIFO Binary Plotter")
        self.resize(1200, 800)

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

        self.acc_x_curve = self.p_acc.plot(pen='r', name="aX")
        self.acc_y_curve = self.p_acc.plot(pen='g', name="aY")
        self.acc_z_curve = self.p_acc.plot(pen='b', name="aZ")

        # Gyroscope plot
        self.p_gyro = self.central.addPlot(row=1, col=0, title="Gyroscope")
        self.p_gyro.showGrid(x=True, y=True)
        self.p_gyro.setLabel("left", "deg/s")
        self.p_gyro.setLabel("bottom", "Time", "s")
        self.p_gyro.setYRange(-300, 300)

        self.gyro_x_curve = self.p_gyro.plot(pen='r', name="gX")
        self.gyro_y_curve = self.p_gyro.plot(pen='g', name="gY")
        self.gyro_z_curve = self.p_gyro.plot(pen='b', name="gZ")

        # Status label
        self.status = QtWidgets.QLabel("Starting...")
        self.statusBar().addWidget(self.status)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(int(1000 / DISPLAY_HZ))

    def update_plots(self):
        with self.buffers.lock:
            if len(self.buffers.t) < 2:
                return

            t = np.array(self.buffers.t, dtype=float)
            ax = np.array(self.buffers.ax, dtype=float)
            ay = np.array(self.buffers.ay, dtype=float)
            az = np.array(self.buffers.az, dtype=float)

            gx = np.array(self.buffers.gx, dtype=float)
            gy = np.array(self.buffers.gy, dtype=float)
            gz = np.array(self.buffers.gz, dtype=float)

            last_count = self.buffers.last_count
            drop_count = self.buffers.drop_count

        # Show only trailing window
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

        self.acc_x_curve.setData(tt, ax)
        self.acc_y_curve.setData(tt, ay)
        self.acc_z_curve.setData(tt, az)

        self.gyro_x_curve.setData(tt, gx)
        self.gyro_y_curve.setData(tt, gy)
        self.gyro_z_curve.setData(tt, gz)

        self.p_acc.setXRange(t_start, t_end, padding=0)
        self.p_gyro.setXRange(t_start, t_end, padding=0)

        approx_rate = 0.0
        if len(tt) >= 2:
            dt = tt[-1] - tt[0]
            if dt > 0:
                approx_rate = (len(tt) - 1) / dt

        self.status.setText(
            f"Last count: {last_count} | Dropped: {drop_count} | "
            f"Window rate: {approx_rate:.1f} Hz"
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