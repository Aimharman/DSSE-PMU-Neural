#define _DEFAULT_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

#define PMU_PACKET_SIZE 542U
#define PMU_HEADER_SIZE 28U
#define PMU_PACKET_MAGIC 0x33554D50UL
#define PMU_PACKET_VERSION 1U
#define PMU_SAMPLE_COUNT 128U

static uint16_t ReadU16(const uint8_t *data)
{
    return (uint16_t)data[0] | ((uint16_t)data[1] << 8U);
}

static uint32_t ReadU32(const uint8_t *data)
{
    return (uint32_t)ReadU16(data) | ((uint32_t)ReadU16(data + 2U) << 16U);
}

static uint64_t ReadU64(const uint8_t *data)
{
    return (uint64_t)ReadU32(data) | ((uint64_t)ReadU32(data + 4U) << 32U);
}

static uint16_t Crc16(const uint8_t *data, size_t length)
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

static int OpenUart(const char *path)
{
    struct termios options;
    int descriptor = open(path, O_RDONLY | O_NOCTTY);

    if ((descriptor < 0) || (tcgetattr(descriptor, &options) != 0))
    {
        return -1;
    }
    cfmakeraw(&options);
    cfsetispeed(&options, B115200);
    cfsetospeed(&options, B115200);
    options.c_cflag |= CLOCAL | CREAD;
    return tcsetattr(descriptor, TCSANOW, &options) == 0 ? descriptor : -1;
}

int main(int argc, char **argv)
{
    const char *device = NULL;
    const char *csv_path = NULL;
    uint8_t packet[PMU_PACKET_SIZE];
    size_t used = 0U;
    uint32_t last_sequence = 0U;
    uint64_t last_timestamp = 0U;
    unsigned long packet_count = 0U;
    unsigned long crc_failures = 0U;
    unsigned long sequence_gaps = 0U;
    unsigned long timestamp_errors = 0U;
    int option;
    int uart;
    FILE *csv;

    while ((option = getopt(argc, argv, "d:o:")) != -1)
    {
        if (option == 'd')
        {
            device = optarg;
        }
        else if (option == 'o')
        {
            csv_path = optarg;
        }
    }
    if ((device == NULL) || (csv_path == NULL))
    {
        fprintf(stderr, "Usage: %s -d /dev/ttyACM0 -o samples.csv\n", argv[0]);
        return 2;
    }
    uart = OpenUart(device);
    csv = fopen(csv_path, "w");
    if ((uart < 0) || (csv == NULL))
    {
        perror("UART or CSV");
        return 1;
    }

    fprintf(csv, "sequence,timestamp_us,sample_index,voltage_raw,current_raw\n");
    for (;;)
    {
        ssize_t received = read(uart, packet + used, PMU_PACKET_SIZE - used);
        if (received <= 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            perror("read");
            break;
        }
        used += (size_t)received;
        while (used >= PMU_PACKET_SIZE)
        {
            uint32_t sequence;
            uint64_t timestamp;
            uint32_t first_sample;
            uint32_t sample;

            if (ReadU32(packet) != PMU_PACKET_MAGIC)
            {
                memmove(packet, packet + 1U, --used);
                continue;
            }
            if ((packet[4] != PMU_PACKET_VERSION) || (packet[7] != 2U) ||
                (ReadU16(packet + 24U) != PMU_SAMPLE_COUNT) ||
                (Crc16(packet, PMU_PACKET_SIZE - 2U) != ReadU16(packet + PMU_PACKET_SIZE - 2U)))
            {
                crc_failures++;
                memmove(packet, packet + 1U, --used);
                continue;
            }

            sequence = ReadU32(packet + 8U);
            timestamp = ReadU64(packet + 12U);
            first_sample = ReadU32(packet + 20U);
            if ((packet_count != 0U) && (sequence != last_sequence + 1U))
            {
                sequence_gaps += sequence - last_sequence - 1U;
            }
            if ((packet_count != 0U) && (timestamp <= last_timestamp))
            {
                timestamp_errors++;
            }
            for (sample = 0U; sample < PMU_SAMPLE_COUNT; sample++)
            {
                fprintf(csv, "%" PRIu32 ",%" PRIu64 ",%" PRIu32 ",%" PRIu16 ",%" PRIu16 "\n",
                        sequence, timestamp + (uint64_t)sample * 1000U, first_sample + sample,
                        ReadU16(packet + PMU_HEADER_SIZE + sample * 4U),
                        ReadU16(packet + PMU_HEADER_SIZE + sample * 4U + 2U));
            }
            fflush(csv);
            packet_count++;
            last_sequence = sequence;
            last_timestamp = timestamp;
            memmove(packet, packet + PMU_PACKET_SIZE, used - PMU_PACKET_SIZE);
            used -= PMU_PACKET_SIZE;
            fprintf(stderr, "packets=%lu crc_failures=%lu sequence_gaps=%lu timestamp_errors=%lu\r",
                    packet_count, crc_failures, sequence_gaps, timestamp_errors);
        }
    }
    fclose(csv);
    close(uart);
    return 0;
}