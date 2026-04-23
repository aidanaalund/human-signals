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

PACKET_FMT = "<BBHIhhhhhh"
PACKET_SIZE = struct.calcsize(PACKET_FMT)

WINDOW_SEC = 5
DISPLAY_HZ = 50
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


def read_line_until(ser, expected_prefix, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print("CTRL:", line)
            if line.startswith(expected_prefix):
                return True
    return False


def send_reset_start(ser):
    ser.write(b"R")
    ser.flush()
    return read_line_until(ser, "OK", timeout=3.0)

def robust_start_handshake(ser, timeout=6.0, retry_interval=0.2):
    """
    Repeatedly send 'R' until Arduino replies 'OK'.
    Ignores whether RDY is missed or arrives late.
    """
    t0 = time.time()
    last_send = 0.0
    line_buf = b""

    while time.time() - t0 < timeout:
        now = time.time()

        # Periodically send reset/start command
        if now - last_send >= retry_interval:
            ser.write(b"R")
            ser.flush()
            last_send = now

        # Read whatever text is available
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            line_buf += chunk

            while b"\n" in line_buf:
                line, line_buf = line_buf.split(b"\n", 1)
                text = line.decode(errors="ignore").strip()
                if text:
                    print("CTRL:", text)
                if text.startswith("OK"):
                    return True

        time.sleep(0.01)

    return False

def serial_worker(buffers: SharedBuffers, stop_event: threading.Event):
    try:
        # timeout used for both readline() and binary reads
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
    except serial.SerialException as e:
        print(f"Could not open serial port: {e}")
        return
    
    # startup

    time.sleep(2.0)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # Send reset/start command
    ser.write(b"R")
    ser.flush()

    # Give Arduino a moment to clear FIFO and restart stream
    time.sleep(0.1)

    # Throw away anything partial/stale after reset
    ser.reset_input_buffer()

    # Clear rate/drop tracking for a fresh run
    with buffers.lock:
        buffers.start_time = time.perf_counter()
        buffers.last_count = None
        buffers.drop_count = 0
        buffers.report_t0 = None
        buffers.report_count0 = None
        buffers.latest_rate_hz = 0.0

        buffers.t.clear()
        buffers.count.clear()

        buffers.ax.clear()
        buffers.ay.clear()
        buffers.az.clear()

        buffers.gx.clear()
        buffers.gy.clear()
        buffers.gz.clear()

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

        if buffers.last_count is None:
            print(f"First packet count after R: {count}")

        now = time.perf_counter()
        t = now - buffers.start_time

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

            buffers.t.append(t)
            buffers.count.append(count)

            buffers.ax.append(ax)
            buffers.ay.append(ay)
            buffers.az.append(az)

            buffers.gx.append(gx)
            buffers.gy.append(gy)
            buffers.gz.append(gz)

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

    # optional stop command on close
    try:
        ser.write(b"S")
        ser.flush()
    except Exception:
        pass

    ser.close()


class IMUWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU Binary Plotter")
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

        self.p_acc = self.central.addPlot(row=0, col=0, title="Accelerometer")
        self.p_acc.showGrid(x=True, y=True)
        self.p_acc.setLabel("left", "g")
        self.p_acc.setLabel("bottom", "Time", "s")
        self.p_acc.setYRange(-2.5, 2.5)

        self.acc_x_curve = self.p_acc.plot(pen='r')
        self.acc_y_curve = self.p_acc.plot(pen='g')
        self.acc_z_curve = self.p_acc.plot(pen='b')

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

    def update_plots(self):
        now = time.perf_counter()
        dt_gui = now - self.gui_last_time
        self.gui_last_time = now
        if dt_gui > 0:
            self.gui_fps = 1.0 / dt_gui

        with self.buffers.lock:
            if len(self.buffers.t) < 2:
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

        self.status.setText(
            f"Last count: {last_count} | Dropped: {drop_count} | "
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