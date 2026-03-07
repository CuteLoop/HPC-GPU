#!/bin/bash
set -euo pipefail

module load cuda11/11.0
module load cmake/3.28.3

export CC=gcc
export CXX=g++

cd /home/u16/joelmaldonado/HPC-GPU/ece569/build_dir

LABS_DIR=/home/u16/joelmaldonado/HPC-GPU/ece569/labs

rm -f CMakeCache.txt Makefile cmake_install.cmake compile_commands.json
rm -rf CMakeFiles

cmake -D CUDA_TOOLKIT_ROOT_DIR=/opt/ohpc/pub/apps/cuda/11.8 "$LABS_DIR"

echo "Configuration complete for ece569/labs. Next: make -j"
