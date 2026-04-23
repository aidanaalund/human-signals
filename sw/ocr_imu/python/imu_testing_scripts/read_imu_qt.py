import sys
import time
import serial
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

PORT = "COM5"        # change this
BAUD = 230400
FS = 100             # sample rate
WINDOW_SEC = 5
N = FS * WINDOW_SEC

ser = serial.Serial(PORT, BAUD, timeout=0.01)
time.sleep(2)

class SerialIMUPlotter:
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)

        self.win = pg.GraphicsLayoutWidget(show=True, title="IMU Live Plot")
        self.win.resize(1000, 600)

        self.plot = self.win.addPlot(title="Accelerometer")
        self.plot.showGrid(x=True, y=True)
        self.plot.setLabel("left", "Acceleration", units="g")
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setYRange(-2, 2)

        self.t = np.linspace(-WINDOW_SEC, 0, N)
        self.ax = np.zeros(N)
        self.ay = np.zeros(N)
        self.az = np.zeros(N)

        self.curve_ax = self.plot.plot(self.t, self.ax, name="aX")
        self.curve_ay = self.plot.plot(self.t, self.ay, name="aY")
        self.curve_az = self.plot.plot(self.t, self.az, name="aZ")

        self.last_update = time.time()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(20)   # 50 Hz GUI refresh

    def update(self):
        got_data = False

        for _ in range(100):
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                break

            parts = line.split(",")
            if len(parts) < 6:
                continue

            try:
                aX, aY, aZ, gX, gY, gZ = map(float, parts[:6])
            except ValueError:
                continue

            self.ax[:-1] = self.ax[1:]
            self.ay[:-1] = self.ay[1:]
            self.az[:-1] = self.az[1:]

            self.ax[-1] = aX
            self.ay[-1] = aY
            self.az[-1] = aZ

            got_data = True

        if got_data:
            self.curve_ax.setData(self.t, self.ax)
            self.curve_ay.setData(self.t, self.ay)
            self.curve_az.setData(self.t, self.az)

    def run(self):
        sys.exit(self.app.exec_())

if __name__ == "__main__":
    plotter = SerialIMUPlotter()
    plotter.run()