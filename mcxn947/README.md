# MCXN947 Base Project

This directory provides the repo-local MCXN947 firmware base built from the validated NXP SDK example layout installed at `/media/ankesh/general/mcux-workspace`.

## Toolchain

- ARM GCC toolchain is already installed and available in PATH.
- The SDK expects `ARMGCC_DIR` to be exported in the environment for CMake-based builds.

Example:

```bash
export ARMGCC_DIR="/media/ankesh/general/ARM/arm-gnu-toolchain-15.2.rel1-x86_64-arm-none-eabi"
export SdkRootDirPath="/media/ankesh/general/mcux-workspace"
```

## Build

```bash
cd mcxn947
mkdir -p build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE="/media/ankesh/general/mcux-workspace/core/tools/cmake_toolchain_files/armgcc.cmake" -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=debug ..
make -j
```

## Flash

The system LinkServer binary is installed and available in PATH:

```bash
LinkServer flash auto load ./mcxn947_base.elf
```

The board-specific flash target may be auto-detected or selected as `MCXN947`/`FRDM-MCXN947` depending on the connected probe. Use `LinkServer devices` to confirm the target identifier before flashing.

## PMU-3 acquisition baseline

The firmware now samples two LPADC channels with one hardware-triggered transaction every 1 ms:

```text
LPTMR0 -> INPUTMUX -> ADC0 command 1 (ADC0_A2) -> command 2 (ADC0_A8) -> FIFO A -> DMA0
```

- ADC0_A2 on P4_23 is the voltage input.
- ADC0_A8 on PTC6 is the current input.
- DMA stores 128 interleaved logical samples per window as `V0, I0, V1, I1, ...`.
- A completed window receives a monotonic timestamp derived from the 1 kHz hardware schedule and is transmitted through the FRDM debug UART (LPUART4) at 115200 baud.
- The red LED toggles when a packet is sent.

The packet is 542 bytes, little-endian: `PMU3` magic, version 1, PMU ID 3, sample type, two channels, sequence, timestamp in microseconds, first sample index, 128 samples, status, 512 bytes of ADC values, then CRC-16/CCITT-FALSE.

The external analogue front end must provide safe, conditioned low-voltage signals. Do not connect mains voltage or current-transducer outputs directly to either ADC pin. The FRDM board header locations and actual scaling must be checked against the board schematic before connecting signals.

## DAC self-test waveform

The firmware also starts an independent DAC0 test source for acquisition validation. It uses a fixed 16-value sine lookup table centered at DAC code 2048 with a peak excursion of 1552 codes. LPTMR1 hardware-triggers DAC0 at 800 Hz, so the table produces a deterministic 50 Hz waveform:

$$
\frac{800\ \mathrm{updates/s}}{16\ \mathrm{table\ entries}} = 50\ \mathrm{Hz}
$$

DAC0 uses DMA0 channel 1 to refill its FIFO. ADC uses LPTMR0 and DMA0 channel 0, so enabling or disabling the waveform generator does not change the PMU acquisition pipeline.

DAC0 output is P4_2, available at FRDM-MCXN947 connector J1-4. For the first physical loop, connect J1-4 through a suitable buffer or conditioning network to a verified accessible ADC voltage input and connect the analogue grounds. The installed SDK identifies ADC0_A2 as P4_23, but this repository does not contain the FRDM schematic needed to prove its header location; verify that point before installing a jumper. Do not assume the shared DAC0/ADC0_A4 pad is an internal loopback.

After wiring, capture a sustained stream with the Pi receiver. The expected logical sample rate is 1000 samples/s/channel, the window duration is 128 ms, and the voltage CSV values should reconstruct the 50 Hz DAC stimulus. Record observed frequency, raw minimum/maximum/mean, packet rate, sequence gaps, CRC failures, and DMA errors before treating the hardware loop as validated.

## Raspberry Pi receiver

Build [pi_receiver/pmu_uart_receiver.c](../pi_receiver/pmu_uart_receiver.c) on the Pi:

```bash
cd pi_receiver
make
./pmu_uart_receiver -d /dev/ttyACM0 -o mcxn947_raw.csv
```

The receiver validates framing, CRC, sequence continuity, and timestamp monotonicity, then records raw ADC codes as CSV. A real acquisition validation requires a sustained capture with the physical analogue front end and Pi serial device connected.
