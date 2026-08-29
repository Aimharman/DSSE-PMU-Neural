#include "pmu_protocol.h"

static void PMU_PutU16(uint8_t *destination, uint16_t value)
{
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8U);
}

static void PMU_PutU32(uint8_t *destination, uint32_t value)
{
    PMU_PutU16(destination, (uint16_t)value);
    PMU_PutU16(destination + 2U, (uint16_t)(value >> 16U));
}

static void PMU_PutU64(uint8_t *destination, uint64_t value)
{
    PMU_PutU32(destination, (uint32_t)value);
    PMU_PutU32(destination + 4U, (uint32_t)(value >> 32U));
}

static uint16_t PMU_Crc16(const uint8_t *data, size_t length)
{
    uint16_t crc = 0xFFFFU;
    size_t index;
    for (index = 0U; index < length; index++)
    {
        uint8_t bit;
        crc ^= (uint16_t)data[index] << 8U;
        for (bit = 0U; bit < 8U; bit++)
        {
            crc = (crc & 0x8000U) ? (uint16_t)((crc << 1U) ^ 0x1021U) : (uint16_t)(crc << 1U);
        }
    }
    return crc;
}

size_t PMU_EncodeSamplePacket(uint8_t *packet, size_t capacity, const pmu_sample_window_t *window)
{
    size_t offset = 0U;
    uint32_t sample_index;

    if ((packet == NULL) || (window == NULL) || (capacity < PMU_PACKET_MAX_SIZE))
    {
        return 0U;
    }
    PMU_PutU32(packet + offset, PMU_PACKET_MAGIC); offset += 4U;
    packet[offset++] = PMU_PACKET_VERSION;
    packet[offset++] = 3U;
    packet[offset++] = 1U;
    packet[offset++] = PMU_CHANNEL_COUNT;
    PMU_PutU32(packet + offset, window->sequence); offset += 4U;
    PMU_PutU64(packet + offset, window->timestamp_us); offset += 8U;
    PMU_PutU32(packet + offset, window->first_sample_index); offset += 4U;
    PMU_PutU16(packet + offset, PMU_SAMPLES_PER_WINDOW); offset += 2U;
    PMU_PutU16(packet + offset, 0U); offset += 2U;
    for (sample_index = 0U; sample_index < PMU_SAMPLES_PER_WINDOW; sample_index++)
    {
        PMU_PutU16(packet + offset, window->samples[sample_index][0]); offset += 2U;
        PMU_PutU16(packet + offset, window->samples[sample_index][1]); offset += 2U;
    }
    PMU_PutU16(packet + offset, PMU_Crc16(packet, offset));
    return offset + 2U;
}