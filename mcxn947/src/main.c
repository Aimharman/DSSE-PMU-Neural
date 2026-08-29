#include "pin_mux.h"
#include "peripherals.h"
#include "board.h"

#define BOARD_LED_GPIO     BOARD_LED_RED_GPIO
#define BOARD_LED_GPIO_PIN BOARD_LED_RED_GPIO_PIN

void SysTick_Handler(void)
{
    GPIO_PortToggle(BOARD_LED_GPIO, 1u << BOARD_LED_GPIO_PIN);
}

int main(void)
{
    CLOCK_EnableClock(kCLOCK_Gpio0);
    BOARD_InitPins();
    BOARD_BootClockFRO12M();
    SysTick_Config(12000000UL);
    LED_RED_INIT(LOGIC_LED_OFF);

    while (1)
    {
    }
}
