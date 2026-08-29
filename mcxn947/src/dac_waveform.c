#include "dac_waveform.h"

#include "fsl_common.h"
#include "fsl_dac.h"
#include "fsl_edma.h"
#include "fsl_inputmux.h"
#include "fsl_inputmux_connections.h"
#include "fsl_lptmr.h"
#include "fsl_spc.h"

#define DAC_DMA_CHANNEL 1U
#define DAC_CLOCK_HZ 12000000U

static const uint32_t s_waveform[DAC_WAVEFORM_TABLE_SIZE] = {
    2048U,  2635U,  3134U,  3477U,  3600U,  3477U,  3134U,  2635U,
    2048U,  1461U,  962U,   619U,   496U,   619U,   962U,   1461U
};

static volatile uint32_t s_dma_transfer_count;

void EDMA_0_CH1_IRQHandler(void)
{
    if ((EDMA_GetChannelStatusFlags(DMA0, DAC_DMA_CHANNEL) & kEDMA_InterruptFlag) != 0U)
    {
        s_dma_transfer_count++;
        EDMA_ClearChannelStatusFlags(DMA0, DAC_DMA_CHANNEL, kEDMA_InterruptFlag);
        EDMA_EnableChannelRequest(DMA0, DAC_DMA_CHANNEL);
    }
}

static void DAC_InitDma(void)
{
    edma_channel_config_t channel_config = {0};
    edma_transfer_config_t transfer_config;

    channel_config.channelRequestSource = kDma0RequestMuxDac0FifoRequest;
    channel_config.channelPreemptionConfig.enablePreemptAbility = true;
    channel_config.protectionLevel = kEDMA_ChannelProtectionLevelUser;
#if !(defined(FSL_FEATURE_EDMA_HAS_NO_CH_SBR_SEC) && FSL_FEATURE_EDMA_HAS_NO_CH_SBR_SEC)
    channel_config.securityLevel = kEDMA_ChannelSecurityLevelNonSecure;
#endif
    EDMA_PrepareTransfer(&transfer_config, (void *)s_waveform, sizeof(s_waveform[0]),
                         (void *)&DAC0->DATA, sizeof(uint32_t), sizeof(uint32_t), sizeof(s_waveform),
                         kEDMA_MemoryToPeripheral);
    transfer_config.srcMajorLoopOffset = -(int32_t)sizeof(s_waveform);
    EDMA_SetTransferConfig(DMA0, DAC_DMA_CHANNEL, &transfer_config, NULL);
    EDMA_InitChannel(DMA0, DAC_DMA_CHANNEL, &channel_config);
    EnableIRQ(EDMA_0_CH1_IRQn);
    EDMA_EnableChannelRequest(DMA0, DAC_DMA_CHANNEL);
}

void DAC_WaveformInit(void)
{
    dac_config_t dac_config;
    lptmr_config_t timer_config;

    CLOCK_SetClkDiv(kCLOCK_DivDac0Clk, 1U);
    CLOCK_AttachClk(kFRO12M_to_DAC0);
    CLOCK_SetupClockCtrl(kCLOCK_FRO12MHZ_ENA);
    SPC_EnableActiveModeAnalogModules(SPC0, kSPC_controlVref | kSPC_controlDac0);

    DAC_GetDefaultConfig(&dac_config);
    dac_config.fifoTriggerMode = kDAC_FIFOTriggerByHardwareMode;
    dac_config.fifoWorkMode = kDAC_FIFOWorkAsNormalMode;
    dac_config.fifoWatermarkLevel = 0U;
    dac_config.referenceVoltageSource = kDAC_ReferenceVoltageSourceAlt1;
    DAC_Init(DAC0, &dac_config);
    DAC_Enable(DAC0, true);

    CLOCK_EnableClock(kCLOCK_InputMux0);
    INPUTMUX_EnableSignal(INPUTMUX0, kINPUTMUX_Dac0FifoRequestToDma0Ch25Ena, true);
    INPUTMUX_AttachSignal(INPUTMUX0, 0U, kINPUTMUX_Lptmr1ToDac0Trigger);
    DAC_InitDma();
    DAC_EnableDMA(DAC0, kDAC_FIFOEmptyDMAEnable, true);

    LPTMR_GetDefaultConfig(&timer_config);
    LPTMR_Init(LPTMR1, &timer_config);
    LPTMR_SetTimerPeriod(LPTMR1, USEC_TO_COUNT(1000000U / DAC_WAVEFORM_UPDATE_RATE_HZ, DAC_CLOCK_HZ));
    LPTMR_StartTimer(LPTMR1);
}

uint32_t DAC_WaveformDmaTransferCount(void)
{
    return s_dma_transfer_count;
}