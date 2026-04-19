import struct
import serial
import time

PORT = "COM5"
BAUD = 230400

FMT = "<BBHhhhhhh"
SIZE = struct.calcsize(FMT)

ACCEL_SCALE_G = 0.000061   # 0.061 mg/LSB at ±2g
GYRO_SCALE_DPS = 0.00875   # 8.75 mdps/LSB at ±245 dps

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2.0)
ser.reset_input_buffer()

def read_exactly(n):
    buf = b""
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

def sync_header():
    while True:
        b = ser.read(1)
        if not b:
            return False
        if b == b"\xAA":
            b2 = ser.read(1)
            if b2 == b"\x55":
                return True

while True:
    if not sync_header():
        continue

    payload = read_exactly(SIZE - 2)
    if payload is None:
        continue

    _, _, count, gx_raw, gy_raw, gz_raw, ax_raw, ay_raw, az_raw = struct.unpack(
        FMT, b"\xAA\x55" + payload
    )

    gx = gx_raw * GYRO_SCALE_DPS
    gy = gy_raw * GYRO_SCALE_DPS
    gz = gz_raw * GYRO_SCALE_DPS

    ax = ax_raw * ACCEL_SCALE_G
    ay = ay_raw * ACCEL_SCALE_G
    az = az_raw * ACCEL_SCALE_G

    print(count, ax, ay, az, gx, gy, gz)