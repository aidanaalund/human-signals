#include <ArduinoBLE.h>
#include <Wire.h>
#include <LSM6DS3.h>

LSM6DS3 myIMU(I2C_MODE, 0x6A);

BLEService imuService("19B10000-E8F2-537E-4F6C-D104768A1214");
BLECharacteristic imuChar(
  "19B10001-E8F2-537E-4F6C-D104768A1214",
  BLERead | BLENotify,
  20
);

const uint32_t SAMPLE_RATE_HZ = 104;
const uint32_t SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;

uint32_t lastSampleTime = 0;
uint16_t sampleCounter = 0;

struct __attribute__((packed)) Packet {
  uint8_t h1;
  uint8_t h2;
  uint16_t count;
  uint32_t t_us;
  int16_t gx_d10;
  int16_t gy_d10;
  int16_t gz_d10;
  int16_t ax_mg;
  int16_t ay_mg;
  int16_t az_mg;
};

static_assert(sizeof(Packet) == 20, "Packet must be 20 bytes");

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin();

  myIMU.settings.gyroEnabled = 1;
  myIMU.settings.gyroRange = 245;
  myIMU.settings.gyroSampleRate = 208;

  myIMU.settings.accelEnabled = 1;
  myIMU.settings.accelRange = 2;
  myIMU.settings.accelSampleRate = 208;

  int rc = myIMU.begin();
  if (rc != 0) {
    while (1) { }
  }

  if (!BLE.begin()) {
    while (1) { }
  }

  BLE.setConnectionInterval(6, 6);  

  BLE.setLocalName("MG24_IMU");
  BLE.setDeviceName("MG24_IMU");
  BLE.setAdvertisedService(imuService);

  imuService.addCharacteristic(imuChar);
  BLE.addService(imuService);

  Packet p = {};
  imuChar.writeValue((uint8_t*)&p, sizeof(p));

  BLE.advertise();

  lastSampleTime = micros();
}

void loop() {
  BLE.poll();

  BLEDevice central = BLE.central();

  if (central) {
    while (central.connected()) {
      BLE.poll();

      while ((uint32_t)(micros() - lastSampleTime) >= SAMPLE_PERIOD_US) {
        lastSampleTime += SAMPLE_PERIOD_US;

        float ax = myIMU.readFloatAccelX();
        float ay = myIMU.readFloatAccelY();
        float az = myIMU.readFloatAccelZ();

        float gx = myIMU.readFloatGyroX();
        float gy = myIMU.readFloatGyroY();
        float gz = myIMU.readFloatGyroZ();

        Packet p;
        p.h1 = 0xAA;
        p.h2 = 0x55;
        p.count = sampleCounter++;
        p.t_us = micros();

        p.ax_mg = (int16_t)(ax * 1000.0f);
        p.ay_mg = (int16_t)(ay * 1000.0f);
        p.az_mg = (int16_t)(az * 1000.0f);

        p.gx_d10 = (int16_t)(gx * 10.0f);
        p.gy_d10 = (int16_t)(gy * 10.0f);
        p.gz_d10 = (int16_t)(gz * 10.0f);

        imuChar.writeValue((uint8_t*)&p, sizeof(p));
      }
    }

    BLE.advertise();
  }
}