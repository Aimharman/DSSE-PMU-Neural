#include "pmu_acquisition.h"
#include "pmu_protocol.h"
#include "dac_waveform.h"
#include "board.h"
#include "fsl_lpuart.h"
#include "pin_mux.h"

#define PMU_UART_BAUD_RATE 115200U

int main(void)
{
    lpuart_config_t uart_config;

    BOARD_BootClockFRO12M();
    BOARD_InitPins();
    LED_RED_INIT(LOGIC_LED_OFF);

    CLOCK_SetClkDiv(kCLOCK_DivFlexcom4Clk, 1U);
    CLOCK_AttachClk(BOARD_DEBUG_UART_CLK_ATTACH);
    LPUART_GetDefaultConfig(&uart_config);
    uart_config.baudRate_Bps = PMU_UART_BAUD_RATE;
    uart_config.enableTx = true;
    uart_config.enableRx = false;
    LPUART_Init(LPUART4, &uart_config, CLOCK_GetLPFlexCommClkFreq(4U));

    DAC_WaveformInit();
    PMU_AcquisitionInit();
    while (1)
    {
        pmu_sample_window_t window;

        if (PMU_AcquisitionTakeWindow(&window))
        {
            uint8_t packet[PMU_PACKET_MAX_SIZE];
            size_t packet_size = PMU_EncodeSamplePacket(packet, sizeof(packet), &window);
            if (packet_size != 0U)
            {
                LPUART_WriteBlocking(LPUART4, packet, packet_size);
                GPIO_PortToggle(BOARD_LED_RED_GPIO, 1U << BOARD_LED_RED_GPIO_PIN);
            }
        }
    }
}
