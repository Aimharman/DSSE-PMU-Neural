rm build -Rv
export ARMGCC_DIR="/media/ankesh/general/ARM/arm-gnu-toolchain-15.2.rel1-x86_64-arm-none-eabi"
export SdkRootDirPath="/media/ankesh/general/mcux-workspace"

cd /home/ankesh/Documents/IIT_Jammu/project_folder/DSSE-PMU-Neural/mcxn947
mkdir -p build
cd build

cmake -DCMAKE_TOOLCHAIN_FILE="/media/ankesh/general/mcux-workspace/core/tools/cmake_toolchain_files/armgcc.cmake" \
      -G "Unix Makefiles" \
      -DCMAKE_BUILD_TYPE=debug \
      ..

make -j"$(nproc)"