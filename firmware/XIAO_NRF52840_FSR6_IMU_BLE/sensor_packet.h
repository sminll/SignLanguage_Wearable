#ifndef SENSOR_PACKET_H
#define SENSOR_PACKET_H

#include <stdint.h>
#include <string.h>
#include <limits>

namespace fsr_ble {

constexpr unsigned kBodySize = 45;
constexpr unsigned kPacketSize = 20;
constexpr unsigned kPacketCount = 3;

struct Sample {
  uint32_t timeMs;
  uint16_t fsr[6];
  float imu[7]; // AX/AY/AZ_g, GX/GY/GZ_dps, TEMP_C.
  bool imuOk;
};

inline void writeUint32Le(uint8_t *output, uint32_t value) {
  for (unsigned i = 0; i < 4; ++i) {
    output[i] = static_cast<uint8_t>(value >> (8 * i));
  }
}

// No native struct layout or host byte-order assumptions enter the wire format.
inline void encodePackets(const Sample &sample, uint32_t sequence,
                          uint8_t packets[kPacketCount][kPacketSize]) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                "The protocol requires IEEE 754 float32");
  uint8_t body[kBodySize];
  writeUint32Le(body, sample.timeMs);
  for (unsigned i = 0; i < 6; ++i) {
    body[4 + i * 2] = static_cast<uint8_t>(sample.fsr[i]);
    body[5 + i * 2] = static_cast<uint8_t>(sample.fsr[i] >> 8);
  }
  for (unsigned i = 0; i < 7; ++i) {
    uint32_t bits = 0x7fc00000u; // Quiet NaN for every invalid IMU field.
    if (sample.imuOk) memcpy(&bits, &sample.imu[i], sizeof(bits));
    writeUint32Le(body + 16 + i * 4, bits);
  }
  body[44] = sample.imuOk ? 1 : 0;

  for (unsigned i = 0; i < kPacketCount; ++i) {
    packets[i][0] = static_cast<uint8_t>(0xa0 + i);
    writeUint32Le(packets[i] + 1, sequence);
    memcpy(packets[i] + 5, body + i * 15, 15);
  }
}

} // namespace fsr_ble

#endif
