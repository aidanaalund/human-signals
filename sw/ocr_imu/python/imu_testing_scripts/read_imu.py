import serial
import time
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

PORT = "COM5"          # change this on Windows
BAUD = 230400
MAX_POINTS = 500       # last 5 seconds at 100 Hz

ser = serial.Serial(PORT, BAUD, timeout=0.1)
time.sleep(2)  # allow board reset if needed

xdata = deque(maxlen=MAX_POINTS)
ax_data = deque(maxlen=MAX_POINTS)
ay_data = deque(maxlen=MAX_POINTS)
az_data = deque(maxlen=MAX_POINTS)

start_time = time.time()

fig, ax = plt.subplots()
line_ax, = ax.plot([], [], label="aX")
line_ay, = ax.plot([], [], label="aY")
line_az, = ax.plot([], [], label="aZ")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Acceleration (g)")
ax.legend()
ax.grid(True)

def update(frame):
    # Read all available lines without blocking too long
    for _ in range(50):
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            break

        parts = line.split(",")
        if len(parts) != 6:
            continue

        try:
            aX, aY, aZ, gX, gY, gZ = map(float, parts)
        except ValueError:
            continue

        t = time.time() - start_time
        xdata.append(t)
        ax_data.append(aX)
        ay_data.append(aY)
        az_data.append(aZ)

    if xdata:
        line_ax.set_data(xdata, ax_data)
        line_ay.set_data(xdata, ay_data)
        line_az.set_data(xdata, az_data)

        ax.set_xlim(max(0, xdata[0]), xdata[-1] + 0.01)

        ymin = min(min(ax_data), min(ay_data), min(az_data)) - 0.1
        ymax = max(max(ax_data), max(ay_data), max(az_data)) + 0.1
        ax.set_ylim(ymin, ymax)

    return line_ax, line_ay, line_az

ani = FuncAnimation(fig, update, interval=50, blit=False)
plt.show()

ser.close()