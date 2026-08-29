/*
 * Minimal pin config mirroring the FRDM-MCXN947 board definitions used by the SDK example.
 */

#include "fsl_common.h"
#include "fsl_port.h"
#include "pin_mux.h"

void BOARD_InitBootPins(void)
{
    BOARD_InitPins();
}

void BOARD_InitPins(void)
{
    CLOCK_EnableClock(kCLOCK_Port0);

    const port_pin_config_t port0_10_pinB12_config = {
        kPORT_PullDisable,
        kPORT_LowPullResistor,
        kPORT_FastSlewRate,
        kPORT_PassiveFilterDisable,
        kPORT_OpenDrainDisable,
        kPORT_LowDriveStrength,
        kPORT_MuxAlt0,
        kPORT_InputBufferEnable,
        kPORT_InputNormal,
        kPORT_UnlockRegister
    };

    PORT_SetPinConfig(PORT0, 10U, &port0_10_pinB12_config);

    const port_pin_config_t port0_2_pinB16_config = {
        kPORT_PullDisable,
        kPORT_LowPullResistor,
        kPORT_FastSlewRate,
        kPORT_PassiveFilterDisable,
        kPORT_OpenDrainDisable,
        kPORT_HighDriveStrength,
        kPORT_MuxAlt1,
        kPORT_InputBufferEnable,
        kPORT_InputNormal,
        kPORT_UnlockRegister
    };

    PORT_SetPinConfig(PORT0, 2U, &port0_2_pinB16_config);
}
