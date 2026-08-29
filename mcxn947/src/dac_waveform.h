#ifndef DAC_WAVEFORM_H
#define DAC_WAVEFORM_H

#include <stdint.h>

#define DAC_WAVEFORM_FREQUENCY_HZ 50U
#define DAC_WAVEFORM_UPDATE_RATE_HZ 800U
#define DAC_WAVEFORM_TABLE_SIZE 16U

void DAC_WaveformInit(void);
uint32_t DAC_WaveformDmaTransferCount(void);

#endif