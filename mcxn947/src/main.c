#include "pmu_acquisition.h"
#include "pmu_protocol.h"
#include "dac_waveform.h"
#include "board.h"
#include "fsl_debug_console.h"
#include "fsl_lpuart.h"
#include "pin_mux.h"

#define PMU_UART_BAUD_RATE 115200U
#define PMU_DEBUG_CONSOLE 1

#if PMU_DEBUG_CONSOLE
static void PMU_PrintWindow(const pmu_sample_window_t *window)
{
    uint32_t sample_index;

    PRINTF("\r\nPMU: window=%u first_sample=%u timestamp_us=%llu adc_dma=%u dac_dma=%u\r\n",
           window->sequence, window->first_sample_index, window->timestamp_us,
           PMU_AcquisitionCompletedWindowCount(), DAC_WaveformDmaTransferCount());
    PRINTF("PMU: ADC V,I first 16 samples:");
    for (sample_index = 0U; sample_index < 16U; sample_index++)
    {
        PRINTF(" %u,%u", window->samples[sample_index][0], window->samples[sample_index][1]);
    }
    PRINTF("\r\n");
}
#endif

int main(void)
{
#if !PMU_DEBUG_CONSOLE
    lpuart_config_t uart_config;
#endif

    BOARD_BootClockFRO12M();
    BOARD_InitPins();
    LED_RED_INIT(LOGIC_LED_OFF);

#if PMU_DEBUG_CONSOLE
    BOARD_InitDebugConsole();
    PRINTF("\r\nPMU: booted, FRO12M selected\r\n");
#else
    CLOCK_SetClkDiv(kCLOCK_DivFlexcom4Clk, 1U);
    CLOCK_AttachClk(BOARD_DEBUG_UART_CLK_ATTACH);
    LPUART_GetDefaultConfig(&uart_config);
    uart_config.baudRate_Bps = PMU_UART_BAUD_RATE;
    uart_config.enableTx = true;
    uart_config.enableRx = false;
    LPUART_Init(LPUART4, &uart_config, CLOCK_GetLPFlexCommClkFreq(4U));
#endif

    PRINTF("PMU: initializing ADC, LPTMR0, and DMA0 channel 0\r\n");
    PMU_AcquisitionInit();
    PRINTF("PMU: ADC acquisition initialized\r\n");
    PRINTF("PMU: initializing DAC0, LPTMR1, and DMA0 channel 1\r\n");
    DAC_WaveformInit();
    PRINTF("PMU: DAC waveform initialized, entering acquisition loop\r\n");
    while (1)
    {
        pmu_sample_window_t window;

        if (PMU_AcquisitionTakeWindow(&window))
        {
#if PMU_DEBUG_CONSOLE
            if ((window.sequence % 8U) == 0U)
            {
                PMU_PrintWindow(&window);
            }
#else
            uint8_t packet[PMU_PACKET_MAX_SIZE];
            size_t packet_size = PMU_EncodeSamplePacket(packet, sizeof(packet), &window);
            if (packet_size != 0U)
            {
                LPUART_WriteBlocking(LPUART4, packet, packet_size);
                GPIO_PortToggle(BOARD_LED_RED_GPIO, 1U << BOARD_LED_RED_GPIO_PIN);
            }
#endif
        }
    }
}
