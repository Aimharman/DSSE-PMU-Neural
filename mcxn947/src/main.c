#include "board.h"
#include "fsl_debug_console.h"
#include "fsl_lpadc.h"
#include "fsl_dac.h"
#include "fsl_spc.h"
#include "pin_mux.h"

#define TEST_ADC_CHANNEL       2U
#define TEST_ADC_COMMAND       1U
#define TEST_ADC_TRIGGER       0U

#define ADC_SAMPLES_PER_LEVEL  16U
#define DAC_SETTLE_TIME_US     100000U
#define BETWEEN_LEVELS_US      500000U

static const uint16_t g_dac_test_values[] =
{
    496U,
    1024U,
    2048U,
    3072U,
    3600U
};


/* --------------------------------------------------------------------------
 * DAC initialization
 * -------------------------------------------------------------------------- */
static void DAC_TestInit(void)
{
    dac_config_t dac_config;

    CLOCK_SetClkDiv(kCLOCK_DivDac0Clk, 1U);
    CLOCK_AttachClk(kFRO12M_to_DAC0);
    CLOCK_SetupClockCtrl(kCLOCK_FRO12MHZ_ENA);

    SPC_EnableActiveModeAnalogModules(SPC0,
                                       kSPC_controlVref |
                                       kSPC_controlDac0);

    DAC_GetDefaultConfig(&dac_config);

    dac_config.fifoTriggerMode = kDAC_FIFOTriggerBySoftwareMode;
    dac_config.fifoWorkMode = kDAC_FIFOWorkAsNormalMode;
    dac_config.fifoWatermarkLevel = 0U;
    dac_config.referenceVoltageSource =
        kDAC_ReferenceVoltageSourceAlt1;

    DAC_Init(DAC0, &dac_config);
    DAC_Enable(DAC0, true);
}


/* --------------------------------------------------------------------------
 * Set DAC output
 * -------------------------------------------------------------------------- */
static void DAC_SetTestValue(uint16_t value)
{
    DAC_SetData(DAC0, value);
    DAC_DoSoftwareTriggerFIFO(DAC0);
}


/* --------------------------------------------------------------------------
 * ADC initialization
 * -------------------------------------------------------------------------- */
static void ADC_TestInit(void)
{
    lpadc_config_t adc_config;
    lpadc_conv_command_config_t command_config;
    lpadc_conv_trigger_config_t trigger_config;

    CLOCK_SetClkDiv(kCLOCK_DivAdc0Clk, 1U);
    CLOCK_AttachClk(kFRO12M_to_ADC0);

    SPC_EnableActiveModeAnalogModules(SPC0,
                                       kSPC_controlVref);

    LPADC_GetDefaultConfig(&adc_config);

    adc_config.enableAnalogPreliminary = true;
    adc_config.powerUpDelay = 0x10U;
    adc_config.referenceVoltageSource =
        kLPADC_ReferenceVoltageAlt3;

    LPADC_Init(ADC0, &adc_config);

    LPADC_DoOffsetCalibration(ADC0);
    LPADC_DoAutoCalibration(ADC0);

    LPADC_GetDefaultConvCommandConfig(&command_config);

    command_config.channelNumber = TEST_ADC_CHANNEL;

    LPADC_SetConvCommandConfig(ADC0,
                                TEST_ADC_COMMAND,
                                &command_config);

    LPADC_GetDefaultConvTriggerConfig(&trigger_config);

    trigger_config.targetCommandId = TEST_ADC_COMMAND;
    trigger_config.enableHardwareTrigger = false;

    LPADC_SetConvTriggerConfig(ADC0,
                               TEST_ADC_TRIGGER,
                               &trigger_config);
}


/* --------------------------------------------------------------------------
 * Read one ADC conversion
 * -------------------------------------------------------------------------- */
static uint16_t ADC_Read(void)
{
    lpadc_conv_result_t result;

    LPADC_DoSoftwareTrigger(ADC0, 1U);

    while (!LPADC_GetConvResult(ADC0, &result, 0U))
    {
    }

    return (uint16_t)result.convValue;
}


/* --------------------------------------------------------------------------
 * Read multiple ADC samples and calculate statistics
 * -------------------------------------------------------------------------- */
static void ADC_ReadStatistics(uint32_t *average,
                               uint16_t *minimum,
                               uint16_t *maximum)
{
    uint32_t sum = 0U;
    uint16_t min_value = 0xFFFFU;
    uint16_t max_value = 0U;

    for (uint32_t i = 0U;
         i < ADC_SAMPLES_PER_LEVEL;
         i++)
    {
        uint16_t value = ADC_Read();

        sum += value;

        if (value < min_value)
        {
            min_value = value;
        }

        if (value > max_value)
        {
            max_value = value;
        }

        /*
         * Small delay between ADC samples.
         */
        SDK_DelayAtLeastUs(1000U,
                           CLOCK_GetFreq(kCLOCK_CoreSysClk));
    }

    *average = sum / ADC_SAMPLES_PER_LEVEL;
    *minimum = min_value;
    *maximum = max_value;
}


/* --------------------------------------------------------------------------
 * Main
 * -------------------------------------------------------------------------- */
int main(void)
{
    BOARD_BootClockFRO12M();
    BOARD_InitPins();
    BOARD_InitDebugConsole();

    PRINTF("\r\n");
    PRINTF("============================================\r\n");
    PRINTF("       MCXN947 DAC -> ADC LOOPBACK TEST\r\n");
    PRINTF("============================================\r\n");
    PRINTF("\r\n");

    PRINTF("ADC channel : %u\r\n", TEST_ADC_CHANNEL);
    PRINTF("Samples     : %u per DAC level\r\n",
           ADC_SAMPLES_PER_LEVEL);
    PRINTF("\r\n");

    PRINTF("Initializing DAC0...\r\n");
    DAC_TestInit();
    PRINTF("DAC0 initialized.\r\n");

    PRINTF("Initializing ADC0...\r\n");
    ADC_TestInit();
    PRINTF("ADC0 initialized.\r\n");

    PRINTF("\r\n");
    PRINTF("Physical connection:\r\n");
    PRINTF("    DAC0 J3-2  --->  ADC0_A2 J8-28\r\n");
    PRINTF("    GND        --->  GND\r\n");
    PRINTF("\r\n");

    PRINTF("Starting test...\r\n");
    PRINTF("\r\n");

    while (1)
    {
        for (uint32_t i = 0U;
             i < sizeof(g_dac_test_values) /
                 sizeof(g_dac_test_values[0]);
             i++)
        {
            uint16_t dac_value = g_dac_test_values[i];

            uint32_t average;
            uint16_t minimum;
            uint16_t maximum;

            /*
             * Set DAC output.
             */
            DAC_SetTestValue(dac_value);

            /*
             * Allow DAC output to settle.
             */
            SDK_DelayAtLeastUs(DAC_SETTLE_TIME_US,
                               CLOCK_GetFreq(kCLOCK_CoreSysClk));

            /*
             * Read ADC multiple times.
             */
            ADC_ReadStatistics(&average,
                               &minimum,
                               &maximum);

            PRINTF("DAC=%4u  ADC avg=%5u  min=%5u  max=%5u\r\n",
                   dac_value,
                   (unsigned)average,
                   (unsigned)minimum,
                   (unsigned)maximum);

            SDK_DelayAtLeastUs(BETWEEN_LEVELS_US,
                               CLOCK_GetFreq(kCLOCK_CoreSysClk));
        }

        PRINTF("\r\n");
    }
}