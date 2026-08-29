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
