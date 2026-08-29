---
name: real_pmu_embedded
description: |
  # DSSE-PMU-Neural — Coding Agent Master Configuration
## Hardware Implementation Roadmap and Current Execution Track

## 1. Purpose

This document is the coding-agent control document for the hardware extension of:

`https://github.com/Aimharman/DSSE-PMU-Neural.git`

Its purpose is to preserve project history, define the intended architecture,
identify the current implementation phase, and prevent the coding agent from silently changing project direction.

The agent must treat this document as the implementation boundary.

The project is being converted from a PMU-simulator/reference workflow into athree-controller physical PMU testbed:

- MCXN947
- STM32F407VET6
- ESP32
- Raspberry Pi as the central host/reference system

The controllers will acquire safe, conditioned analog signals using their built-in ADCs and operate as individual PMU nodes.

---

# 2. Project History / Existing Baseline

The current repository is the software/reference baseline.

The documented v5 architecture combines:

1. Neural fault detection
2. WLS state estimation
3. Chi-squared bad-data detection
4. Decision fusion

The documented neural fault classes are:

- NORMAL
- BAD_DATA
- SYNC
- CLOCK_DRIFT

The current neural/reference workflow uses a 128-sample window at approximately
1 kHz, giving a 128 ms processing window.

The current Python/reference implementation remains the golden reference.

The hardware implementation must reproduce the required data semantics rather
than redesign the existing DSSE/neural system.

---

# 3. Important Hardware Status

## MCXN947

MCXN947 has already been successfully brought up.

Verified:

- ARM GCC build works
- project builds
- firmware can be flashed
- Blinky runs on the board

Therefore the MCXN947 is the FIRST hardware platform for implementation.

The next implementation phase is NOT STM32.

The next implementation phase is:

> MCXN947 physical PMU acquisition.

---

## STM32F407VET6

STM32 is the SECOND hardware platform.

Important toolchain constraint:

- ARM GCC 15.2 is already installed and available in `PATH`
- There is NO STM32CubeMX requirement
- There is NO `.ioc`-based workflow
- The STM32 project should use a command-line / native-source build system
- OpenOCD will be configured for programming/debugging

The coding agent must NOT introduce CubeMX or CubeIDE.

The STM32 firmware will be developed directly using the available ARM GCC
toolchain and the appropriate STM32 device/startup/linker/support files.

---

## ESP32

ESP32 is the THIRD hardware platform.

It will be implemented after the two ARM-M-class PMU paths are established,
unless a concrete project dependency makes a different order necessary.

---

# 4. Overall Target Architecture

Final intended topology:

                    SAFE ANALOG SOURCES
                       /     |     \
                      /      |      \
                     v       v       v
                 MCXN947   STM32   ESP32
                  PMU-3    PMU-1   PMU-2
                     \       |       /
                      \      |      /
                       +-----+-----+
                             |
                       PMU data streams
                             |
                             v
                        Raspberry Pi
                  +----------------------+
                  | Acquisition / logger |
                  | Time alignment       |
                  | WLS                  |
                  | Chi-squared          |
                  | Neural reference     |
                  | Decision fusion      |
                  +----------------------+

Final advanced target:

                       MCXN947
                    +------------+
                    | ADC / DMA  |
                    |     |      |
                    | preprocess |
                    |     |      |
                    | neural NN  |
                    |     |      |
                    | classifier |
                    +-----+------+
                          |
                          v
                     PMU/fault data
                          |
                          v
                    Raspberry Pi

MCXN947 is intended to become the edge-AI platform after physical acquisition
is proven.

Raspberry Pi remains the supervisory and golden-reference platform.

---

# 5. Hardware Development Order

The project SHALL proceed in this order unless a documented technical reason
requires a controlled deviation.

## Phase 1 — MCXN947 PMU-3 Bring-up

CURRENT PHASE.

1. ADC
2. timer-triggered sampling
3. eDMA
4. two-channel acquisition
5. sample buffering
6. timestamping
7. packet generation
8. UART/initial transport
9. Raspberry Pi receiver
10. acquisition validation

---

## Phase 2 — STM32F407VET6 PMU-1

After MCXN947 acquisition is proven:

1. direct ARM GCC project
2. startup/system initialization
3. ADC
4. timer-triggered sampling
5. DMA
6. two-channel acquisition
7. timestamping
8. packet generation
9. OpenOCD flashing/debugging
10. Raspberry Pi interoperability

---

## Phase 3 — ESP32 PMU-2

1. ADC1
2. calibrated acquisition
3. buffering
4. timestamping
5. common PMU packet format
6. host interoperability

---

## Phase 4 — Multi-PMU Synchronization

1. common synchronization signal
2. timestamp capture
3. sample counters
4. sequence counters
5. Raspberry Pi time alignment
6. measurable synchronization error

---

## Phase 5 — Physical Fault Injection

Implement repeatable physical/controlled versions of:

- NORMAL
- BAD_DATA
- SYNC
- CLOCK_DRIFT

---

## Phase 6 — MCXN947 Embedded Neural Inference

1. inspect reference model
2. freeze input/features
3. select portable embedded representation
4. implement inference
5. compare against Python golden reference
6. benchmark latency/RAM/flash
7. run inference on live PMU windows

---

## Phase 7 — Complete Fusion Demonstration

Final system:

physical PMUs
   |
   +--> neural stream
   |
   +--> classical DSSE/WLS
   |
   +--> chi-squared
   |
   +--> decision fusion

with MCXN947 performing the embedded neural workload.

---

# 6. CURRENT TASK — MCXN947 PHYSICAL PMU

## Objective

Produce the first verified physical PMU data path on the MCXN947.

Target:

SAFE ANALOG INPUT
       |
       v
MCXN947 ADC
       |
       v
hardware-triggered sampling
       |
       v
eDMA buffer
       |
       v
timestamp/sample counter
       |
       v
PMU packet
       |
       v
UART
       |
       v
Raspberry Pi

This task does NOT include neural deployment.

---

# 7. MCXN947 Initial Configuration

Use the board/project configuration already present in the repository or
workspace.

Do not invent unnecessary project infrastructure.

Initial acquisition target:

- two analog inputs
- logical channel 0 = voltage
- logical channel 1 = current
- effective sampling rate = 1 kHz per channel
- hardware timer trigger
- ADC
- eDMA
- circular/ping-pong buffering
- 128-sample logical processing window
- monotonic timestamp
- sample/window counter
- sequence number
- UART as initial transport

The exact ADC instance, channel numbers, trigger source and pins must be
determined from the actual MCXN947 board/project configuration.

Do not invent pin assignments.

---

# 8. MCXN947 Firmware Structure

First inspect the existing MCXN947 Blinky project that was successfully built
and flashed.

Reuse its:

- compiler configuration
- linker setup
- startup files
- CMSIS/device support
- board support
- build system
- flashing workflow

Extend that known-good project.

Do NOT create a separate unrelated firmware project unless the existing
structure makes extension technically impractical.

Preferred logical modules:

```text
mcxn947/
├── README.md
├── src/
│   ├── main.c
│   ├── adc.c
│   ├── adc.h
│   ├── dma.c
│   ├── dma.h
│   ├── timer.c
│   ├── timer.h
│   ├── pmu.c
│   ├── pmu.h
│   ├── timestamp.c
│   ├── timestamp.h
│   ├── protocol.c
│   └── protocol.h
└── ...
```

This is a logical recommendation.

If the existing MCUXpresso SDK / project organization differs, follow the
existing organization rather than forcing this exact tree.

---

# 9. MCXN947 Acquisition Rules

The acquisition path must be hardware driven.

Preferred:

Timer
  -> ADC trigger
  -> ADC conversion
  -> eDMA
  -> buffer
  -> processing flag/event

Do NOT use an ADC polling loop as the primary implementation.

Do NOT put expensive work inside DMA interrupts/callbacks.

DMA callbacks may:
- update indices
- update counters
- set flags
- mark buffers ready

Higher-level code performs:
- packetization
- calculations
- UART transmission

---

# 10. Buffer Strategy

Use a robust DMA buffer.

A logical sample is:

```text
voltage
current
```

Keep raw ADC samples in integer form initially.

Preferred initial raw representation:

`uint16_t`

The exact interleaving/layout must be explicitly documented.

Example:

```text
V0 I0 V1 I1 V2 I2 ...
```

or separate arrays.

Choose one and keep it identical throughout the MCXN947 host decoder.

The 128-sample neural/reference window is a logical application window, not
permission to redesign the acquisition mechanism.

---

# 11. Sampling Validation

Target:

`1000 samples/second/channel`

Two channels:

`2000 channel values/second`

Logical window:

`128 samples`

Window duration:

`128 ms`

The firmware must allow actual sample timing to be measured.

Document:

- MCU/core clock used
- timer source
- timer configuration
- ADC trigger rate
- effective measured sample rate
- buffer size
- DMA transfer layout

Do not claim 1 kHz merely because the timer calculation says 1 kHz.

---

# 12. Analog Input Safety

NEVER connect mains voltage/current directly to an MCU ADC.

The initial laboratory source must be safe and conditioned.

Conceptual signal:

`ADC = offset + amplitude * sin(2*pi*f*t + phase)`

ADC input must remain within the allowed electrical range.

Expected external analog front-end functions:

- scaling
- bias/offset
- protection
- filtering

The first validation should preferably use a controlled low-voltage waveform
from a signal generator or equivalent laboratory source.

---

# 13. PMU Measurement Scope — CURRENT PHASE

Initially acquire and transport raw voltage/current samples.

Do NOT start with:

- phasor algorithms
- WLS
- chi-squared
- neural inference
- complicated PMU calculations

Once raw acquisition is proven, later phases can add:

- RMS
- frequency
- phase
- active power
- reactive power
- other required measurement quantities

The final measurement vector must be derived from the existing DSSE model.

Do not invent neural features.

---

# 14. Timestamp and Counters

Each PMU data block must carry at least:

- PMU ID
- timestamp
- sample/window counter
- packet sequence number
- sample count
- status

The timestamp must be generated from a local hardware/software time base,
not inferred from UART arrival time.

Timestamping must be designed so later hardware synchronization can be added
without redesigning the entire packet protocol.

---

# 15. Initial PMU Packet

For the first milestone, use a versioned binary packet carrying:

- protocol version
- PMU ID
- packet type
- sequence number
- timestamp
- sample/window counter
- channel count
- sample count
- voltage samples
- current samples
- status flags
- CRC

Requirements:

- explicit field widths
- explicit endianness
- explicit scaling
- CRC
- versioning
- no undocumented compiler struct packing

A temporary debug output may exist on a separate debug interface.

Do not mix arbitrary text logging into the binary PMU data stream.

---

# 16. Raspberry Pi Receiver

Create or extend a clearly separated host receiver.

The receiver must initially:

1. receive UART data
2. identify packet boundaries
3. validate version
4. validate CRC
5. decode fields
6. check sequence continuity
7. record errors
8. save raw samples
9. display basic acquisition statistics

Do NOT alter WLS/neural/fusion code for this task.

The receiver is the first independent verification that the MCXN947 PMU
output is correct.

---

# 17. MCXN947 First Acceptance Test

Phase 1 is NOT complete merely because firmware builds or flashes.

It is complete only when all are true:

[ ] Existing MCXN947 project still builds

[ ] New PMU firmware flashes successfully

[ ] ADC captures two analog channels

[ ] Timer triggers acquisition

[ ] eDMA continuously moves ADC samples

[ ] Voltage/current channel ordering is verified

[ ] Effective sample rate is measured near 1 kHz/channel

[ ] 128-sample windows can be identified

[ ] Timestamp is monotonic

[ ] Sequence counter increments correctly

[ ] No unexplained DMA overrun occurs

[ ] PMU packet is versioned

[ ] CRC is generated correctly

[ ] Raspberry Pi receives packets

[ ] Raspberry Pi validates CRC

[ ] Raspberry Pi reconstructs the sample stream

[ ] Raw samples can be saved

[ ] Extended acquisition shows no unexplained sample loss

Only then may the MCXN947 PMU phase be marked COMPLETE.

---

# 18. MCXN947 Neural Work — EXPLICITLY DEFERRED

Do not start neural deployment during the first acquisition milestone.

After PMU acquisition is stable, inspect the Python reference model and determine:

- exact feature vector
- exact feature order
- input dimensions
- normalization
- model architecture
- activations
- weights/parameters
- class order
- confidence calculation
- windowing

Then select an embedded representation suitable for MCXN947.

The Python/joblib model must NOT simply be assumed to execute directly on the
MCXN947.

The Python implementation remains the golden reference.

---

# 19. STM32 Implementation — FUTURE PHASE

STM32 is deliberately postponed until MCXN947 acquisition succeeds.

Constraints:

- ARM GCC 15.2 is already in PATH
- no CubeMX
- no CubeIDE dependency
- direct source/build approach
- OpenOCD for flash/debug
- reuse standard STM32 device/startup/linker support as appropriate

Before coding STM32:

1. inspect the existing repository for STM32 files
2. inspect board-specific information available
3. establish exact MCU clock/startup assumptions
4. establish ADC pin/channel assignments
5. establish timer and DMA resources
6. configure OpenOCD
7. verify a minimal build/flash/debug path
8. then implement the same PMU logical interface as MCXN947

The STM32 implementation should reproduce the MCXN947 PMU data contract,
not create a second incompatible protocol.

---

# 20. ESP32 Implementation — FUTURE PHASE

ESP32 is third in the hardware order.

Initial direction:

- use ADC1
- calibrated acquisition
- two channels
- approximately 1 kHz/channel
- buffered acquisition
- timestamp
- common packet protocol

Do not begin ESP32 work during the current MCXN947 phase.

---

# 21. Synchronization — FUTURE PHASE

Synchronization is important because SYNC and CLOCK_DRIFT are explicit neural
fault classes.

Final target:

                 HARDWARE SYNC
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
       MCXN947      STM32       ESP32
          |           |           |
     timestamps   timestamps  timestamps
          |           |           |
          +-----------+-----------+
                      |
                      v
                 Raspberry Pi

At minimum retain:

- local timestamp
- sample counter
- packet sequence
- synchronization event/counter

Do not finalize a software-only synchronization architecture without
measurement and justification.

---

# 22. Physical Fault Testing — FUTURE PHASE

The physical system must eventually reproduce:

## NORMAL

All PMUs correctly measured and synchronized.

## BAD_DATA

Controlled measurement corruption/outlier on one PMU.

## SYNC

Controlled timing/phase synchronization error.

## CLOCK_DRIFT

Controlled timestamp/sample-clock drift.

Fault magnitudes must be measurable and recorded.

The physical test cases should map to the existing software/reference scenario
definitions as closely as practical.

---

# 23. Golden Reference

For embedded neural validation:

Same recorded input
      |
      +--> Python reference
      |
      +--> MCXN947 implementation

Compare:

- features
- normalized features
- intermediate values where useful
- outputs
- class
- confidence

The goal is numerical/behavioral equivalence within a documented tolerance,
not merely "the firmware produces a plausible class."

---

# 24. Repository Modification Rules

Before changing anything:

```bash
git status
git branch --show-current
git log -5 --oneline
```

Inspect first.

Preserve user changes.

Do not:
- reset
- rebase
- force-push
- discard unrelated modifications

Do not create large unrelated refactors.

Keep hardware code isolated from the existing Python/reference logic.

Do not modify the neural training/model code unless a later, explicitly
approved model-porting phase requires it.

---

# 25. Agent Scope for CURRENT TASK

The current agent is authorized to modify/create only what is required for:

MCXN947 physical PMU acquisition and its minimal Raspberry Pi receiver.

This includes:

- MCXN947 ADC
- MCXN947 timer trigger
- MCXN947 eDMA
- MCXN947 buffering
- MCXN947 timestamp/counters
- MCXN947 PMU protocol
- MCXN947 UART transport
- minimal Raspberry Pi packet decoder/logger
- documentation/tests needed to validate the above

The current agent is NOT authorized to implement:

- STM32 firmware
- ESP32 firmware
- neural inference
- WLS
- chi-squared
- decision fusion
- final synchronization hardware
- complex analog hardware design

unless explicitly requested in a later phase.

---

# 26. Documentation Required for MCXN947

Create/update the MCXN947 hardware README in the appropriate existing
hardware directory.

It must document:

- board/project used
- build command
- flash command
- MCU clock assumptions
- ADC instance/channel configuration
- GPIO/pin assumptions
- timer source/configuration
- eDMA configuration
- buffer organization
- sampling rate
- timestamp method
- UART settings
- packet format
- host receiver
- validation procedure
- measured results
- known limitations

Do not document assumptions as measured facts.

---

# 27. Build / Flash Requirements

The agent must first use the existing MCXN947 build/flash workflow that has
already been proven with Blinky.

Before changing build infrastructure, inspect and reuse:

- existing Makefile/CMake/build files
- ARM GCC settings
- linker script
- startup code
- SDK/device support
- existing flash/debug configuration

Do not replace a known-good build system without necessity.

---

# 28. Progress Reporting

At the end of each coding iteration report:

## Current Phase
Example:
`Phase 1 — MCXN947 Physical PMU`

## Completed
- ...

## Tested
- ...

## Measured
- actual sample rate
- buffer behavior
- packet rate
- errors

## Files Changed
- ...

## Commit
- hash/message, if committed

## Remaining
- ...

## Next Single Milestone
- ...

Use "implemented" and "tested" separately.

Compilation alone does not equal completion.

---

# 29. Deviation Control

If implementation reveals a need to change the architecture, do not silently
change it.

Report:

1. Existing plan
2. Problem discovered
3. Proposed change
4. Why the change is required
5. Impact on later phases
6. Validation required

Do not change:
- sampling strategy
- neural window
- class definitions
- PMU data semantics
- controller roles
- project architecture

without explicit justification.

---

# 30. Immediate Agent Instructions

The first action is repository/project inspection.

Run:

```bash
git status
git branch --show-current
git log -5 --oneline
find . -maxdepth 3 -type f | sort
```

Then identify the already-working MCXN947 Blinky project.

Determine:

- build system
- linker/startup files
- SDK/device support
- source layout
- current clock setup
- board definitions
- current UART possibilities
- ADC/eDMA examples or infrastructure already available

Do not start by writing new files.

After inspection, implement the smallest possible step:

> MCXN947 timer-triggered two-channel ADC acquisition using eDMA, with a
> verified 1 kHz effective sample rate.

Then add packetization and Raspberry Pi reception.

---

# 31. Project "Do Not Deviate" Rule

The project must always have ONE current milestone.

Current milestone:

> **MCXN947 PMU acquisition → Raspberry Pi verified raw data stream**

The agent must not jump ahead to neural inference merely because the MCU can
support it.

The correct order is:

MCXN947 acquisition
    ->
MCXN947 host communication
    ->
STM32 acquisition
    ->
ESP32 acquisition
    ->
multi-PMU synchronization
    ->
physical fault injection
    ->
MCXN947 neural inference
    ->
WLS / chi-squared / fusion integration
    ->
complete demonstration

This ordering is deliberate and must be preserved unless a documented
technical reason requires a change.