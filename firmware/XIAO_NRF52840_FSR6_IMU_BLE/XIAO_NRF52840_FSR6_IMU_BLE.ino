#include <Arduino.h>
#include <ArduinoBLE.h>
#include <Wire.h>
#include <nrf_gpio.h>

#include "sensor_packet.h"

#if !defined(ARDUINO_SEEED_XIAO_NRF52840_SENSE)
#error "Select Seeeduino mbed: XIAO nRF52840 Sense (No Updates)."
#endif

// ArduinoBLE 2.1.0 / Seeed mbed 2.9.3. USB is optional diagnostics only.
// FSR1..6 -> A0..A5. The onboard IMU uses Wire1, not external A4/A5.
constexpr uint8_t FSR_PINS[] = {A0, A1, A2, A3, A4, A5};
constexpr size_t FSR_COUNT = sizeof(FSR_PINS) / sizeof(FSR_PINS[0]);
constexpr uint8_t IMU_ADDRESS = 0x6A;
constexpr uint32_t SAMPLE_PERIOD_MS = 100;
constexpr char DEVICE_NAME[] = "XIAO-FSR-IMU";
BLEService sensorService("19B10010-E8F2-537E-4F6C-D104768A1214");
BLECharacteristic sensorCharacteristic(
    "19B10011-E8F2-537E-4F6C-D104768A1214", BLENotify,
    fsr_ble::kPacketSize, true); // Fixed 20 bytes; default ATT MTU is sufficient.
bool imuReady = false;
bool bleReady = false;
uint32_t sequence = 0;
uint32_t lastSampleMs = 0;

bool readRegisters(uint8_t reg, uint8_t *buffer, size_t count) {
  Wire1.beginTransmission(IMU_ADDRESS);
  Wire1.write(reg);
  if (Wire1.endTransmission(false) != 0) return false;
  if (Wire1.requestFrom(IMU_ADDRESS, count, true) != count) return false;
  for (size_t i = 0; i < count; ++i) buffer[i] = Wire1.read();
  return true;
}

bool writeRegister(uint8_t reg, uint8_t value) {
  Wire1.beginTransmission(IMU_ADDRESS);
  Wire1.write(reg);
  Wire1.write(value);
  return Wire1.endTransmission() == 0;
}

bool initializeImu() {
  pinMode(PIN_LSM6DS3TR_C_POWER, OUTPUT);
  // Match Seeed's LSM6DS3 library: high-drive power output on P1.08.
  nrf_gpio_cfg(NRF_GPIO_PIN_MAP(1, 8), NRF_GPIO_PIN_DIR_OUTPUT,
               NRF_GPIO_PIN_INPUT_DISCONNECT, NRF_GPIO_PIN_NOPULL,
               NRF_GPIO_PIN_H0H1, NRF_GPIO_PIN_NOSENSE);
  digitalWrite(PIN_LSM6DS3TR_C_POWER, HIGH);
  delay(20);
  Wire1.begin();
  Wire1.setClock(100000);

  uint8_t id = 0;
  if (!readRegisters(0x0F, &id, 1)) {
    if (SERIAL_PORT_MONITOR) SERIAL_PORT_MONITOR.println("# IMU WHO_AM_I read failed");
    return false;
  }
  if (SERIAL_PORT_MONITOR) {
    SERIAL_PORT_MONITOR.print("# IMU WHO_AM_I=0x");
    SERIAL_PORT_MONITOR.println(id, HEX);
  }
  if (id != 0x6A) {
    if (SERIAL_PORT_MONITOR) SERIAL_PORT_MONITOR.println("# Expected LSM6DS3TR-C WHO_AM_I=0x6A");
    return false;
  }

  if (!writeRegister(0x12, 0x01)) return false; // Software reset.
  bool resetDone = false;
  for (uint8_t attempt = 0; attempt < 20; ++attempt) {
    uint8_t control = 0;
    if (!readRegisters(0x12, &control, 1)) return false;
    if ((control & 0x01) == 0) {
      resetDone = true;
      break;
    }
    delay(5);
  }
  if (!resetDone) return false;

  const uint8_t configuration[][2] = {
      {0x12, 0x44}, // CTRL3_C: BDU + automatic register increment.
      {0x10, 0x3C}, // CTRL1_XL: 52 Hz, +/-8 g.
      {0x11, 0x34}, // CTRL2_G: 52 Hz, +/-500 deg/s.
  };
  for (const auto &setting : configuration) {
    uint8_t actual = 0;
    if (!writeRegister(setting[0], setting[1]) ||
        !readRegisters(setting[0], &actual, 1) || actual != setting[1]) {
      if (SERIAL_PORT_MONITOR) SERIAL_PORT_MONITOR.println("# IMU register configuration failed");
      return false;
    }
  }
  delay(100);
  if (SERIAL_PORT_MONITOR) SERIAL_PORT_MONITOR.println("# LSM6DS3TR-C register configuration verified");
  return true;
}

bool readImuValues(float values[7]) {
  uint8_t status = 0;
  if (!readRegisters(0x1E, &status, 1)) return false;
  if ((status & 0x03) != 0x03) return false; // No new accel/gyro pair yet.
  uint8_t buffer[14];
  if (!readRegisters(0x20, buffer, sizeof(buffer))) return false;
  int16_t raw[7]; // Temperature, gyro XYZ, acceleration XYZ; little endian.
  for (size_t i = 0; i < 7; ++i) {
    raw[i] = static_cast<int16_t>(static_cast<uint16_t>(buffer[2 * i]) |
              (static_cast<uint16_t>(buffer[2 * i + 1]) << 8));
  }
  for (size_t i = 0; i < 3; ++i) {
    values[i] = raw[i + 4] * 0.000244f;
    values[i + 3] = raw[i + 1] * 0.0175f;
  }
  values[6] = raw[0] / 256.0f + 25.0f;
  return true;
}

void setup() {
  SERIAL_PORT_MONITOR.begin(115200); // No wait for a USB host.
  analogReadResolution(12);
  analogReference(AR_VDD);
  for (size_t i = 0; i < FSR_COUNT; ++i) pinMode(FSR_PINS[i], INPUT);
  imuReady = initializeImu();
  if (!imuReady && SERIAL_PORT_MONITOR) {
    SERIAL_PORT_MONITOR.println("# IMU initialization failed; BLE IMU fields = nan");
  }

  if (!BLE.begin()) {
    if (SERIAL_PORT_MONITOR) SERIAL_PORT_MONITOR.println("# BLE start failed");
    return;
  }
  BLE.setLocalName(DEVICE_NAME);
  BLE.setDeviceName(DEVICE_NAME);
  BLE.setAdvertisedService(sensorService);
  sensorService.addCharacteristic(sensorCharacteristic);
  BLE.addService(sensorService);
  if (!BLE.advertise()) {
    if (SERIAL_PORT_MONITOR) SERIAL_PORT_MONITOR.println("# BLE advertising failed");
    BLE.end();
    return;
  }
  bleReady = true;
  lastSampleMs = millis();
  if (SERIAL_PORT_MONITOR) SERIAL_PORT_MONITOR.println("# BLE advertising: XIAO-FSR-IMU");
}

void loop() {
  if (!bleReady) {
    delay(1);
    return;
  }
  BLE.poll();
  const uint32_t now = millis();
  if (static_cast<uint32_t>(now - lastSampleMs) < SAMPLE_PERIOD_MS) {
    delay(1);
    return;
  }
  lastSampleMs = now;
  fsr_ble::Sample sample = {};
  sample.timeMs = now;
  for (size_t i = 0; i < FSR_COUNT; ++i) sample.fsr[i] = analogRead(FSR_PINS[i]);
  sample.imuOk = imuReady && readImuValues(sample.imu);
  const uint32_t frameSequence = sequence++; // Advance even when this sample is lost.
  if (!BLE.connected() || !sensorCharacteristic.subscribed()) return;

  uint8_t packets[fsr_ble::kPacketCount][fsr_ble::kPacketSize];
  fsr_ble::encodePackets(sample, frameSequence, packets);
  for (unsigned i = 0; i < fsr_ble::kPacketCount; ++i) {
    BLE.poll();
    if (!BLE.connected() || !sensorCharacteristic.subscribed() ||
        sensorCharacteristic.writeValue(packets[i], fsr_ble::kPacketSize) !=
            static_cast<int>(fsr_ble::kPacketSize)) {
      // Abandon the remaining fragments. The receiver discards the partial frame.
      // ArduinoBLE reports local writes, not acknowledged over-the-air delivery.
      if (SERIAL_PORT_MONITOR) SERIAL_PORT_MONITOR.println("# BLE partial frame discarded");
      break;
    }
  }
}
