rm build -Rv
export ARMGCC_DIR="/home/ankesh/ARM/arm-gnu-toolchain-15.2.rel1-aarch64-arm-none-eabi"
export SdkRootDirPath="/home/ankesh/mcux-workspace"

cd /home/ankesh/Documents/Development-directory/DSSE-Main-Build/DSSE-PMU-Neural/mcxn947
mkdir -p build
cd build

cmake -DCMAKE_TOOLCHAIN_FILE="/home/ankesh/mcux-workspace/core/tools/cmake_toolchain_files/armgcc.cmake" \
      -G "Unix Makefiles" \
      -DCMAKE_BUILD_TYPE=debug \
      ..

make -j"$(nproc)"