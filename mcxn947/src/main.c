#include "board.h"
#include "fsl_common.h"
#include "fsl_debug_console.h"
#include "fsl_gpio.h"
#include "pin_mux.h"
#include "pmu_acquisition.h"
#include "pmu_protocol.h"

/* 1 = run only button test, 0 = run PMU acquisition mode */
#define APP_BUTTON_TEST_MODE 0U
/* 1 = enable PMU text diagnostics, 0 = pure binary packet stream on UART */
#define APP_PMU_TEXT_DIAG_MODE 0U

#if APP_PMU_TEXT_DIAG_MODE
#define APP_PMU_LOG(...) PRINTF(__VA_ARGS__)
#else
#define APP_PMU_LOG(...)
#endif

static bool APP_IsButtonPressed(GPIO_Type *gpio, uint32_t pin)
{
    return GPIO_PinRead(gpio, pin) == 0U;
}

int main(void)
{
    gpio_pin_config_t sw2_config = {kGPIO_DigitalInput, 0U};
#if APP_BUTTON_TEST_MODE
    gpio_pin_config_t sw3_config = {kGPIO_DigitalInput, 0U};
    bool sw2_last;
    bool sw3_last;
    uint32_t heartbeat = 0U;
#else
    pmu_sample_window_t window;
    uint8_t packet[PMU_PACKET_MAX_SIZE];
    bool button_was_pressed = false;
    bool send_request = false;
    bool waiting_window_logged = false;
    uint32_t heartbeat_ticks = 0U;
    uint32_t last_completed_windows = 0U;
    uint32_t button_press_count = 0U;
    uint32_t packet_send_count = 0U;
#endif

    BOARD_BootClockFRO12M();
    CLOCK_EnableClock(kCLOCK_Gpio0);
    BOARD_InitPins();
    BOARD_InitDebugConsole();
#if APP_BUTTON_TEST_MODE
    PRINTF("\r\n[APP] Program start\r\n");
    PRINTF("[APP] Init SW2 GPIO input...\r\n");
#else
    APP_PMU_LOG("\r\n[APP] Program start\r\n");
    APP_PMU_LOG("[APP] Init SW2 GPIO input...\r\n");
#endif
    GPIO_PinInit(BOARD_SW2_GPIO, BOARD_SW2_GPIO_PIN, &sw2_config);

#if APP_BUTTON_TEST_MODE
    PRINTF("[APP] Init SW3 GPIO input...\r\n");
    GPIO_PinInit(BOARD_SW3_GPIO, BOARD_SW3_GPIO_PIN, &sw3_config);

    sw2_last = APP_IsButtonPressed(BOARD_SW2_GPIO, BOARD_SW2_GPIO_PIN);
    sw3_last = APP_IsButtonPressed(BOARD_SW3_GPIO, BOARD_SW3_GPIO_PIN);

    PRINTF("[APP] BUTTON TEST MODE ENABLED\r\n");
    PRINTF("[APP] SW2=%u SW3=%u (pressed=1, released=0)\r\n",
           sw2_last ? 1U : 0U,
           sw3_last ? 1U : 0U);

    while (1)
    {
        bool sw2_now = APP_IsButtonPressed(BOARD_SW2_GPIO, BOARD_SW2_GPIO_PIN);
        bool sw3_now = APP_IsButtonPressed(BOARD_SW3_GPIO, BOARD_SW3_GPIO_PIN);

        if (sw2_now != sw2_last)
        {
            sw2_last = sw2_now;
            PRINTF("[APP] SW2 change -> %u\r\n", sw2_now ? 1U : 0U);
        }

        if (sw3_now != sw3_last)
        {
            sw3_last = sw3_now;
            PRINTF("[APP] SW3 change -> %u\r\n", sw3_now ? 1U : 0U);
        }

        heartbeat++;
        if (heartbeat >= 200U)
        {
            heartbeat = 0U;
            PRINTF("[APP] heartbeat SW2=%u SW3=%u\r\n",
                   sw2_now ? 1U : 0U,
                   sw3_now ? 1U : 0U);
        }

        SDK_DelayAtLeastUs(5000U, CLOCK_GetFreq(kCLOCK_CoreSysClk));
    }

#else
    APP_PMU_LOG("[PMU] Init acquisition...\r\n");
    PMU_AcquisitionInit();
    APP_PMU_LOG("[PMU] Acquisition init done. Press SW2 to request one packet.\r\n");

    while (1)
    {
        bool button_pressed = APP_IsButtonPressed(BOARD_SW2_GPIO, BOARD_SW2_GPIO_PIN);
        uint32_t completed_windows = PMU_AcquisitionCompletedWindowCount();

        if (completed_windows != last_completed_windows)
        {
            APP_PMU_LOG("[PMU] DMA windows completed=%u\r\n", (unsigned int)completed_windows);
            last_completed_windows = completed_windows;
        }

        if (button_pressed && !button_was_pressed)
        {
            button_press_count++;
            send_request = true;
            waiting_window_logged = false;
                 APP_PMU_LOG("[PMU] SW2 press detected. press_count=%u completed_windows=%u\r\n",
                   (unsigned int)button_press_count,
                   (unsigned int)completed_windows);
        }
        button_was_pressed = button_pressed;

        if (send_request && !waiting_window_logged)
        {
            waiting_window_logged = true;
            APP_PMU_LOG("[PMU] Waiting for next window...\r\n");
        }

        if (send_request && PMU_AcquisitionTakeWindow(&window))
        {
            size_t index;
            size_t packet_size = PMU_EncodeSamplePacket(packet, sizeof(packet), &window);

            APP_PMU_LOG("[PMU] Window acquired. seq=%u first_sample=%u timestamp_us=%u\r\n",
                   (unsigned int)window.sequence,
                   (unsigned int)window.first_sample_index,
                   (unsigned int)window.timestamp_us);

            if (packet_size > 0U)
            {
                APP_PMU_LOG("[PMU] Sending packet bytes=%u\r\n", (unsigned int)packet_size);
                for (index = 0U; index < packet_size; index++)
                {
                    (void)PUTCHAR((int)packet[index]);
                }
                send_request = false;
                packet_send_count++;
                APP_PMU_LOG("\r\n[PMU] Packet send complete. sent_count=%u\r\n", (unsigned int)packet_send_count);
            }
            else
            {
                APP_PMU_LOG("[PMU] Packet encode failed (size=0).\r\n");
            }
        }

        heartbeat_ticks++;
        if (heartbeat_ticks >= 1000U)
        {
            heartbeat_ticks = 0U;
                 APP_PMU_LOG("[PMU] heartbeat: completed=%u send_request=%u button=%u\r\n",
                   (unsigned int)PMU_AcquisitionCompletedWindowCount(),
                   send_request ? 1U : 0U,
                   button_pressed ? 1U : 0U);
        }

        SDK_DelayAtLeastUs(1000U, CLOCK_GetFreq(kCLOCK_CoreSysClk));
    }
#endif
}