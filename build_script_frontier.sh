#!/bin/bash

set -e
set -x

######## User Configurations ########
# Source directory
caesar_src_dir=.
# Build directory
caesar_build_dir=./build
caesar_install_dir=$(pwd)/install

export CXX=$(which g++)
export CC=$(which gcc)

mkdir -p ${caesar_build_dir}
mkdir -p ${caesar_install_dir}

cmake -S ${caesar_src_dir} -B ${caesar_build_dir} \
    -DCMAKE_CXX_COMPILER=$(which g++) \
    -DCMAKE_C_COMPILER=$(which gcc) \
    -DCMAKE_PREFIX_PATH="$ROCM_PATH/lib/cmake" \
    -DTorch_DIR="$HOME/miniforge3/envs/py311torch_rocm/lib/python3.11/site-packages/torch/share/cmake/Torch" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON \
    -DCMAKE_INSTALL_PREFIX=${caesar_install_dir}

cmake --build ${caesar_build_dir} -- -j 8
cmake --install ${caesar_build_dir}