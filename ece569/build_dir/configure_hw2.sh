#!/bin/bash
set -euo pipefail

module load cuda11/11.0
module load cmake/3.28.3

export CC=gcc

cd /home/u16/joelmaldonado/HPC-GPU/ece569/build_dir

rm -f CMakeCache.txt Makefile cmake_install.cmake compile_commands.json
rm -rf CMakeFiles

cmake -D CUDA_TOOLKIT_ROOT_DIR=/opt/ohpc/pub/apps/cuda/11.8 ../../Assignment2

echo "Configuration complete. Next step: run 'make -j' on HPC when ready."
