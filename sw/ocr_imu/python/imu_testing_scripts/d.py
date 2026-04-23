import sys
import time
import struct
import threading
from collections import deque

import serial
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

PORT = "COM5"
BAUD = 230400

FMT = "<BBHIhhhhhh"   # header, count, t_us, gx gy gz ax ay az
SIZE = struct.calcsize(FMT)

WINDOW_SEC = 5.0
DISPLAY_HZ = 50
MAX_SAMPLES = 1200

ACCEL_G_PER_LSB = 0.000061      # 0.061 mg/LSB at ±2g
GYRO_DPS_PER_LSB = 0.00875      # 8.75 mdps/LSB at 245 dps

class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.count = deque(maxlen=MAX_SAMPLES)
        self.t_mcu = deque(maxlen=MAX_SAMPLES)
        self.t_host = deque(maxlen=MAX_SAMPLES)

        self.ax = deque(maxlen=MAX_SAMPLES)
        self.ay = deque(maxlen=MAX_SAMPLES)
        self.az = deque(maxlen=MAX_SAMPLES)

        self.gx = deque(maxlen=MAX_SAMPLES)
        self.gy = deque(maxlen=MAX_SAMPLES)
        self.gz = deque(maxlen=MAX_SAMPLES)

        self.drop_count = 0
        self.last_count = None

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

def serial_worker(shared, stop_event):
    ser = serial.Serial(PORT, BAUD, timeout=0.5)
    time.sleep(2.0)
    ser.reset_input_buffer()

    while not stop_event.is_set():
        if not sync_header(ser):
            continue

        payload = read_exactly(ser, SIZE - 2)
        if payload is None:
            continue

        try:
            _, _, count, t_us, gx_raw, gy_raw, gz_raw, ax_raw, ay_raw, az_raw = \
                struct.unpack(FMT, b"\xAA\x55" + payload)
        except struct.error:
            continue

        t_host = time.perf_counter()
        t_mcu = t_us * 1e-6

        gx = gx_raw * GYRO_DPS_PER_LSB
        gy = gy_raw * GYRO_DPS_PER_LSB
        gz = gz_raw * GYRO_DPS_PER_LSB

        ax = ax_raw * ACCEL_G_PER_LSB
        ay = ay_raw * ACCEL_G_PER_LSB
        az = az_raw * ACCEL_G_PER_LSB

        with shared.lock:
            if shared.last_count is not None:
                expected = (shared.last_count + 1) & 0xFFFF
                if count != expected:
                    shared.drop_count += ((count - expected) & 0xFFFF)

            shared.last_count = count
            shared.count.append(count)
            shared.t_mcu.append(t_mcu)
            shared.t_host.append(t_host)

            shared.ax.append(ax)
            shared.ay.append(ay)
            shared.az.append(az)

            shared.gx.append(gx)
            shared.gy.append(gy)
            shared.gz.append(gz)

    ser.close()

class Win(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU Burst-Read Plotter")
        self.resize(1200, 800)

        self.shared = Shared()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=serial_worker, args=(self.shared, self.stop_event), daemon=True)
        self.worker.start()

        self.central = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.central)

        self.p_acc = self.central.addPlot(row=0, col=0, title="Accelerometer")
        self.p_acc.showGrid(x=True, y=True)
        self.p_acc.setLabel("left", "g")
        self.p_acc.setLabel("bottom", "MCU time", "s")
        self.p_acc.setYRange(-2.2, 2.2)

        self.accx = self.p_acc.plot(pen='r')
        self.accy = self.p_acc.plot(pen='g')
        self.accz = self.p_acc.plot(pen='b')

        self.p_gyro = self.central.addPlot(row=1, col=0, title="Gyroscope")
        self.p_gyro.showGrid(x=True, y=True)
        self.p_gyro.setLabel("left", "deg/s")
        self.p_gyro.setLabel("bottom", "MCU time", "s")
        self.p_gyro.setYRange(-300, 300)

        self.gyrx = self.p_gyro.plot(pen='r')
        self.gyry = self.p_gyro.plot(pen='g')
        self.gyrz = self.p_gyro.plot(pen='b')

        self.status = QtWidgets.QLabel("Starting...")
        self.statusBar().addWidget(self.status)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(int(1000 / DISPLAY_HZ))

    def update_plots(self):
        with self.shared.lock:
            if len(self.shared.t_mcu) < 2:
                return

            count = np.array(self.shared.count, dtype=np.uint32)
            t_mcu = np.array(self.shared.t_mcu, dtype=float)
            t_host = np.array(self.shared.t_host, dtype=float)

            ax = np.array(self.shared.ax, dtype=float)
            ay = np.array(self.shared.ay, dtype=float)
            az = np.array(self.shared.az, dtype=float)

            gx = np.array(self.shared.gx, dtype=float)
            gy = np.array(self.shared.gy, dtype=float)
            gz = np.array(self.shared.gz, dtype=float)

            dropped = self.shared.drop_count

        t_end = t_mcu[-1]
        t_start = max(0.0, t_end - WINDOW_SEC)
        mask = t_mcu >= t_start

        tt = t_mcu[mask]
        ax = ax[mask]; ay = ay[mask]; az = az[mask]
        gx = gx[mask]; gy = gy[mask]; gz = gz[mask]

        self.accx.setData(tt, ax)
        self.accy.setData(tt, ay)
        self.accz.setData(tt, az)

        self.gyrx.setData(tt, gx)
        self.gyry.setData(tt, gy)
        self.gyrz.setData(tt, gz)

        self.p_acc.setXRange(t_start, t_end, padding=0)
        self.p_gyro.setXRange(t_start, t_end, padding=0)

        # True received packet rate over host time
        host_dt = t_host[-1] - t_host[0]
        rx_rate = (len(t_host) - 1) / host_dt if host_dt > 0 else 0.0

        # True sample rate according to MCU timestamps
        mcu_dt = t_mcu[-1] - t_mcu[0]
        mcu_rate = (len(t_mcu) - 1) / mcu_dt if mcu_dt > 0 else 0.0

        # Counter-based receive rate over current buffer
        dc = (int(count[-1]) - int(count[0])) & 0xFFFF
        count_rate = dc / host_dt if host_dt > 0 else 0.0

        self.status.setText(
            f"count={int(count[-1])} | dropped={dropped} | "
            f"MCU rate={mcu_rate:.1f} Hz | RX rate={rx_rate:.1f} Hz | Count/host={count_rate:.1f} Hz"
        )

    def closeEvent(self, event):
        self.stop_event.set()
        self.worker.join(timeout=1.0)
        event.accept()

app = QtWidgets.QApplication(sys.argv)
w = Win()
w.show()
sys.exit(app.exec_())