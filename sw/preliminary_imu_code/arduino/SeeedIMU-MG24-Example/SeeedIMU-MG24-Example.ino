// #include <LSM6DS3.h>
// #include <Wire.h>

// LSM6DS3 myIMU(I2C_MODE, 0x6A);

// struct __attribute__((packed)) SamplePacket {
//   uint8_t h1;
//   uint8_t h2;
//   uint16_t count;
//   int16_t gx_d10;
//   int16_t gy_d10;
//   int16_t gz_d10;
//   int16_t ax_mg;
//   int16_t ay_mg;
//   int16_t az_mg;
// };

// uint16_t sampleCounter = 0;

// void setup() {
//   Serial.begin(230400);
//   Wire.begin();

//   // Basic sensor settings
//   myIMU.settings.gyroEnabled = 1;
//   myIMU.settings.gyroRange = 245;          // dps
//   myIMU.settings.gyroSampleRate = 104;     // Hz

//   myIMU.settings.accelEnabled = 1;
//   myIMU.settings.accelRange = 2;           // g
//   myIMU.settings.accelSampleRate = 104;    // Hz

//   // FIFO settings
//   myIMU.settings.gyroFifoEnabled = 1;
//   myIMU.settings.gyroFifoDecimation = 1;

//   myIMU.settings.accelFifoEnabled = 1;
//   myIMU.settings.accelFifoDecimation = 1;

//   myIMU.settings.fifoThreshold = 10;       // watermark
//   myIMU.settings.fifoSampleRate = 104;     // Hz
//   myIMU.settings.fifoModeWord = 6;         // continuous mode

//   if (myIMU.begin() != 0) {
//     while (1) { }
//   }

//   myIMU.fifoBegin();
//   myIMU.fifoClear();

//   delay(50);
// }

// void loop() {
//   // Bit 15 = watermark reached in the example library usage
//   if ((myIMU.fifoGetStatus() & 0x8000) == 0) {
//     return;
//   }

//   // Drain until empty bit is set
//   while ((myIMU.fifoGetStatus() & 0x1000) == 0) {
//     // Order follows the SparkFun/Seeed FIFO example:
//     // gx, gy, gz, ax, ay, az
//     float gx = myIMU.calcGyro(myIMU.fifoRead());
//     float gy = myIMU.calcGyro(myIMU.fifoRead());
//     float gz = myIMU.calcGyro(myIMU.fifoRead());

//     float ax = myIMU.calcAccel(myIMU.fifoRead());
//     float ay = myIMU.calcAccel(myIMU.fifoRead());
//     float az = myIMU.calcAccel(myIMU.fifoRead());

//     SamplePacket p;
//     p.h1 = 0xAA;
//     p.h2 = 0x55;
//     p.count = sampleCounter++;

//     p.gx_d10 = (int16_t)(gx * 10.0f);
//     p.gy_d10 = (int16_t)(gy * 10.0f);
//     p.gz_d10 = (int16_t)(gz * 10.0f);

//     p.ax_mg = (int16_t)(ax * 1000.0f);
//     p.ay_mg = (int16_t)(ay * 1000.0f);
//     p.az_mg = (int16_t)(az * 1000.0f);

//     Serial.write((uint8_t*)&p, sizeof(p));
//   }
// }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <Wire.h>
#include <LSM6DS3.h>

LSM6DS3 myIMU(I2C_MODE, 0x6A);

const uint32_t SAMPLE_RATE_HZ = 52;
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

void setup() {
  Serial.begin(230400);
  Wire.begin();

  myIMU.settings.gyroEnabled = 1;
  myIMU.settings.gyroRange = 245;
  myIMU.settings.gyroSampleRate = 104;

  myIMU.settings.accelEnabled = 1;
  myIMU.settings.accelRange = 2;
  myIMU.settings.accelSampleRate = 104;

  int rc = myIMU.begin();
  if (rc != 0) {
    while (1) { }
  }

  lastSampleTime = micros();
}

void loop() {
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

    Serial.write((uint8_t*)&p, sizeof(p));
  }
}

// ^ WORKS AT 52Hz Perfectly!
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// #include <Wire.h>
// #include <LSM6DS3.h>

// LSM6DS3 myIMU(I2C_MODE, 0x6A);

// struct __attribute__((packed)) Packet {
//   uint8_t h1;
//   uint8_t h2;
//   uint16_t count;
//   uint32_t t_us;
//   int16_t gx_d10;
//   int16_t gy_d10;
//   int16_t gz_d10;
//   int16_t ax_mg;
//   int16_t ay_mg;
//   int16_t az_mg;
// };

// uint16_t sampleCounter = 0;
// bool streaming = false;

// static void resetAndStartFIFO() {
//   myIMU.fifoClear();
//   delay(20);
//   sampleCounter = 0;
//   streaming = true;
// }

// static void stopStreaming() {
//   streaming = false;
// }

// void setup() {
//   Serial.begin(230400);
//   Wire.begin();

//   myIMU.settings.gyroEnabled = 1;
//   myIMU.settings.gyroRange = 245;
//   myIMU.settings.gyroSampleRate = 104;

//   myIMU.settings.accelEnabled = 1;
//   myIMU.settings.accelRange = 2;
//   myIMU.settings.accelSampleRate = 104;

//   myIMU.settings.gyroFifoEnabled = 1;
//   myIMU.settings.gyroFifoDecimation = 1;

//   myIMU.settings.accelFifoEnabled = 1;
//   myIMU.settings.accelFifoDecimation = 1;

//   myIMU.settings.fifoThreshold = 10;   // watermark
//   myIMU.settings.fifoSampleRate = 104;
//   myIMU.settings.fifoModeWord = 6;     // continuous mode

//   int rc = myIMU.begin();
//   if (rc != 0) {
//     while (1) { }
//   }

//   myIMU.fifoBegin();
//   myIMU.fifoClear();

//   // Safe to send text before binary streaming starts
//   // Serial.println("RDY");
// }

// void loop() {
//   // Handle simple commands
//   while (Serial.available() > 0) {
//     char c = (char)Serial.read();

//     if (c == 'R') {
//       resetAndStartFIFO();
//       delay(20);   // optional small separation before binary starts
//     } else if (c == 'S') {
//       stopStreaming();
//     }
//   }

//   if (!streaming) {
//     return;
//   }

//   uint16_t status = myIMU.fifoGetStatus();

//   // Bit 14 is commonly the FIFO overrun bit in this library family
//   if (status & 0x4000) {
//     // If overrun happens, re-clear and restart cleanly
//     myIMU.fifoClear();
//     delay(10);
//     sampleCounter = 0;
//     return;
//   }

//   // Wait until watermark reached
//   if ((status & 0x8000) == 0) {
//     return;
//   }

//   // Drain FIFO until empty
//   while ((myIMU.fifoGetStatus() & 0x1000) == 0) {
//     float gx = myIMU.calcGyro(myIMU.fifoRead());
//     float gy = myIMU.calcGyro(myIMU.fifoRead());
//     float gz = myIMU.calcGyro(myIMU.fifoRead());

//     float ax = myIMU.calcAccel(myIMU.fifoRead());
//     float ay = myIMU.calcAccel(myIMU.fifoRead());
//     float az = myIMU.calcAccel(myIMU.fifoRead());

//     Packet p;
//     p.h1 = 0xAA;
//     p.h2 = 0x55;
//     p.count = sampleCounter++;
//     p.t_us = micros();

//     p.gx_d10 = (int16_t)(gx * 10.0f);
//     p.gy_d10 = (int16_t)(gy * 10.0f);
//     p.gz_d10 = (int16_t)(gz * 10.0f);

//     p.ax_mg = (int16_t)(ax * 1000.0f);
//     p.ay_mg = (int16_t)(ay * 1000.0f);
//     p.az_mg = (int16_t)(az * 1000.0f);

//     Serial.write((uint8_t*)&p, sizeof(p));
//   }
// }