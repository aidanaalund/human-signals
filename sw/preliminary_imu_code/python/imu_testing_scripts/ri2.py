import serial
import struct
import time

PORT = "COM5"
BAUD = 230400

FMT = "<BBHhhhhhh"
SIZE = struct.calcsize(FMT)

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

last_count = None

while True:
    if not sync_header():
        print("waiting for header...")
        continue

    rest = read_exactly(SIZE - 2)
    if rest is None:
        print("timeout reading payload")
        continue

    packet = struct.unpack(FMT, b"\xAA\x55" + rest)
    _, _, count, gx_d10, gy_d10, gz_d10, ax_mg, ay_mg, az_mg = packet

    gx = gx_d10 / 10.0
    gy = gy_d10 / 10.0
    gz = gz_d10 / 10.0
    ax = ax_mg / 1000.0
    ay = ay_mg / 1000.0
    az = az_mg / 1000.0

    if last_count is not None and ((last_count + 1) & 0xFFFF) != count:
        print(f"dropped packet(s): prev={last_count}, now={count}")

    last_count = count
    print(count, ax, ay, az, gx, gy, gz)