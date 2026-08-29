#ifndef PMU_ACQUISITION_H
#define PMU_ACQUISITION_H

#include <stdbool.h>
#include <stdint.h>

#define PMU_CHANNEL_COUNT 2U
#define PMU_SAMPLES_PER_WINDOW 128U

typedef struct
{
    uint32_t sequence;
    uint64_t timestamp_us;
    uint32_t first_sample_index;
    uint16_t samples[PMU_SAMPLES_PER_WINDOW][PMU_CHANNEL_COUNT];
} pmu_sample_window_t;

void PMU_AcquisitionInit(void);
bool PMU_AcquisitionTakeWindow(pmu_sample_window_t *window);

#endif