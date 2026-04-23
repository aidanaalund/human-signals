import serial
import struct

PORT = "COM5"
BAUD = 230400
PACKET_FMT = "<BBhhhhhh"   # little-endian: 2 uint8 + 6 int16
PACKET_SIZE = struct.calcsize(PACKET_FMT)

ser = serial.Serial(PORT, BAUD, timeout=1)

def read_exactly(n):
    buf = b""
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            raise TimeoutError("Serial timeout")
        buf += chunk
    return buf

def sync_to_header():
    while True:
        b = read_exactly(1)
        if b == b'\xAA':
            b2 = read_exactly(1)
            if b2 == b'\x55':
                return

while True:
    sync_to_header()
    payload = read_exactly(PACKET_SIZE - 2)
    packet = struct.unpack(PACKET_FMT, b"\xAA\x55" + payload)

    _, _, ax_mg, ay_mg, az_mg, gx_d10, gy_d10, gz_d10 = packet

    ax = ax_mg / 1000.0
    ay = ay_mg / 1000.0
    az = az_mg / 1000.0
    gx = gx_d10 / 10.0
    gy = gy_d10 / 10.0
    gz = gz_d10 / 10.0

    print(ax, ay, az, gx, gy, gz)