#ifndef PMU_PROTOCOL_H
#define PMU_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#include "pmu_acquisition.h"

#define PMU_PACKET_MAGIC 0x33554D50UL
#define PMU_PACKET_VERSION 1U
#define PMU_PACKET_MAX_SIZE 542U

size_t PMU_EncodeSamplePacket(uint8_t *packet, size_t capacity, const pmu_sample_window_t *window);

#endif