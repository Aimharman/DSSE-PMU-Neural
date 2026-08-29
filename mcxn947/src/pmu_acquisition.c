#include "pmu_acquisition.h"

#include "board.h"
#include "fsl_common.h"
#include "fsl_edma.h"
#include "fsl_inputmux.h"
#include "fsl_inputmux_connections.h"
#include "fsl_lpadc.h"
#include "fsl_lptmr.h"
#include "fsl_spc.h"

#define PMU_SAMPLE_RATE_HZ 1000U
#define PMU_DMA_CHANNEL 0U
#define PMU_LPADC_VOLTAGE_CHANNEL 2U
#define PMU_LPADC_CURRENT_CHANNEL 8U

AT_NONCACHEABLE_SECTION_ALIGN_INIT(
    static uint32_t s_dma_results[PMU_SAMPLES_PER_WINDOW * PMU_CHANNEL_COUNT], 32U) = {0U};

static volatile uint32_t s_completed_windows;
static uint32_t s_consumed_windows;

void EDMA_0_CH0_IRQHandler(void)
{
    if ((EDMA_GetChannelStatusFlags(DMA0, PMU_DMA_CHANNEL) & kEDMA_InterruptFlag) != 0U)
    {
        s_completed_windows++;
        EDMA_ClearChannelStatusFlags(DMA0, PMU_DMA_CHANNEL, kEDMA_InterruptFlag);
        EDMA_EnableChannelRequest(DMA0, PMU_DMA_CHANNEL);
    }
}

static void PMU_InitAdc(void)
{
    lpadc_config_t adc_config;
    lpadc_conv_command_config_t command_config;
    lpadc_conv_trigger_config_t trigger_config;

    CLOCK_SetClkDiv(kCLOCK_DivAdc0Clk, 1U);
    CLOCK_AttachClk(kFRO12M_to_ADC0);
    SPC_EnableActiveModeAnalogModules(SPC0, kSPC_controlVref);

    LPADC_GetDefaultConfig(&adc_config);
    adc_config.enableAnalogPreliminary = true;
    adc_config.powerUpDelay = 0x10U;
    adc_config.referenceVoltageSource = kLPADC_ReferenceVoltageAlt3;
    LPADC_Init(ADC0, &adc_config);
    LPADC_DoOffsetCalibration(ADC0);
    LPADC_DoAutoCalibration(ADC0);

    LPADC_GetDefaultConvCommandConfig(&command_config);
    command_config.channelNumber = PMU_LPADC_VOLTAGE_CHANNEL;
    command_config.chainedNextCommandNumber = 2U;
    LPADC_SetConvCommandConfig(ADC0, 1U, &command_config);

    LPADC_GetDefaultConvCommandConfig(&command_config);
    command_config.channelNumber = PMU_LPADC_CURRENT_CHANNEL;
    LPADC_SetConvCommandConfig(ADC0, 2U, &command_config);

    LPADC_GetDefaultConvTriggerConfig(&trigger_config);
    trigger_config.targetCommandId = 1U;
    trigger_config.enableHardwareTrigger = true;
    LPADC_SetConvTriggerConfig(ADC0, 0U, &trigger_config);
    LPADC_EnableFIFO0WatermarkDMA(ADC0, true);
}

static void PMU_InitDma(void)
{
    edma_config_t dma_config;
    edma_channel_config_t channel_config = {0};
    edma_transfer_config_t transfer_config;
    void *fifo = (void *)&ADC0->RESFIFO[0U];

    EDMA_GetDefaultConfig(&dma_config);
    EDMA_Init(DMA0, &dma_config);
    channel_config.channelRequestSource = kDma0RequestMuxAdc0FifoARequest;
    channel_config.channelPreemptionConfig.enablePreemptAbility = true;
    channel_config.protectionLevel = kEDMA_ChannelProtectionLevelUser;
#if !(defined(FSL_FEATURE_EDMA_HAS_NO_CH_SBR_SEC) && FSL_FEATURE_EDMA_HAS_NO_CH_SBR_SEC)
    channel_config.securityLevel = kEDMA_ChannelSecurityLevelNonSecure;
#endif
    EDMA_PrepareTransfer(&transfer_config, fifo, sizeof(uint32_t), s_dma_results,
                         sizeof(s_dma_results[0]), sizeof(s_dma_results[0]), sizeof(s_dma_results),
                         kEDMA_PeripheralToMemory);
    transfer_config.dstMajorLoopOffset = -(int32_t)sizeof(s_dma_results);
    EDMA_SetTransferConfig(DMA0, PMU_DMA_CHANNEL, &transfer_config, NULL);
    EDMA_InitChannel(DMA0, PMU_DMA_CHANNEL, &channel_config);
    EnableIRQ(EDMA_0_CH0_IRQn);
    EDMA_EnableChannelRequest(DMA0, PMU_DMA_CHANNEL);
}

static void PMU_InitTriggerTimer(void)
{
    lptmr_config_t timer_config;

    CLOCK_EnableClock(kCLOCK_InputMux0);
    CLOCK_SetupClockCtrl(kCLOCK_FRO12MHZ_ENA);
    INPUTMUX_EnableSignal(INPUTMUX0, kINPUTMUX_Adc0FifoARequestToDma0Ch21Ena, true);
    INPUTMUX_AttachSignal(INPUTMUX0, 0U, kINPUTMUX_Lptmr0ToAdc0Trigger);

    LPTMR_GetDefaultConfig(&timer_config);
    LPTMR_Init(LPTMR0, &timer_config);
    LPTMR_SetTimerPeriod(LPTMR0, USEC_TO_COUNT(1000000U / PMU_SAMPLE_RATE_HZ, 12000000U));
}

void PMU_AcquisitionInit(void)
{
    PMU_InitTriggerTimer();
    PMU_InitAdc();
    PMU_InitDma();
    LPTMR_StartTimer(LPTMR0);
}

bool PMU_AcquisitionTakeWindow(pmu_sample_window_t *window)
{
    uint32_t completed_windows = s_completed_windows;
    uint32_t sample_index;

    if ((window == NULL) || (completed_windows == s_consumed_windows))
    {
        return false;
    }

    s_consumed_windows = completed_windows;
    window->sequence = completed_windows - 1U;
    window->first_sample_index = window->sequence * PMU_SAMPLES_PER_WINDOW;
    window->timestamp_us = (uint64_t)window->first_sample_index * (1000000U / PMU_SAMPLE_RATE_HZ);
    for (sample_index = 0U; sample_index < PMU_SAMPLES_PER_WINDOW; sample_index++)
    {
        window->samples[sample_index][0] = (uint16_t)((s_dma_results[sample_index * PMU_CHANNEL_COUNT] & 0x7FFFU) >> 3U);
        window->samples[sample_index][1] = (uint16_t)((s_dma_results[sample_index * PMU_CHANNEL_COUNT + 1U] & 0x7FFFU) >> 3U);
    }
    return true;
}

uint32_t PMU_AcquisitionCompletedWindowCount(void)
{
    return s_completed_windows;
}