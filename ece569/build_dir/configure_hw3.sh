#!/bin/bash
set -euo pipefail

module load cuda11/11.0
module load cmake/3.28.3

export CC=gcc
export CXX=g++

LABS_DIR=/home/u16/joelmaldonado/HPC-GPU/ece569/labs
BUILD_DIR=/home/u16/joelmaldonado/HPC-GPU/ece569/build_dir

cd "$BUILD_DIR"

rm -f CMakeCache.txt Makefile cmake_install.cmake compile_commands.json
rm -rf CMakeFiles

cmake -D CUDA_TOOLKIT_ROOT_DIR=/opt/ohpc/pub/apps/cuda/11.8 "$LABS_DIR"

echo "Configuration complete. Next step: run 'make -j' on HPC when ready."