#!/bin/bash

set -x

#conda create --name GCG python=3.10
#conda activate GCG
#conda install -c conda-forge libstdcxx-ng

#sudo apt install libhwloc-dev libpciaccess-dev libboost-dev

#git clone https://github.com/pytorch/pytorch
#cd pytorch
#
#git submodule sync
#git submodule update --init --recursive
##
##
#
#conda install cmake ninja
##pip install -r requirements.txt
#
#
##export USE_CUDA=1
##export _GLIBCXX_USE_CXX11_ABI=0
#rm -rf build
#
#export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
#python -m pip install --no-build-isolation -v -e .
#exit
####
#cd ..


PYTORCH_ROOT=$(pwd)/pytorch
NCCL_INC=${PYTORCH_ROOT}/build/nccl/include

cd GCGScheduler

pip install -r requirements.txt

rm -rf build
mkdir build
cd build


-DCMAKE_PREFIX_PATH=`python3 -c'import torch;print(torch.utils.cmake_prefix_path)'` \
-DRAY_DIR=`python -c'import os;import ray;print(os.path.abspath(os.path.dirname(ray.__file__)))'` \


-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_CXX_FLAGS="-I${NCCL_INC}" \

-DWITH_CUDA=On \

cmake \
-DCMAKE_BUILD_TYPE=Release \
-DWITH_CANN=On \
-Dpybind11_DIR=`python -c "import pybind11; print(pybind11.get_cmake_dir())"` \
..

make -j VERBOSE=1


