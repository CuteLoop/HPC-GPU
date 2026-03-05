#!/bin/bash
set -euo pipefail

module load cuda11/11.0
module load cmake/3.28.3

export CC=gcc
export CXX=g++

cd /home/u16/joelmaldonado/HPC-GPU/ece569/build_dir

ASSIGNMENT3_DIR=/home/u16/joelmaldonado/HPC-GPU/Assignment3
LIBWB_TARGET="$ASSIGNMENT3_DIR/libwb"
LIBWB_SOURCE=/home/u16/joelmaldonado/HPC-GPU/ece569/labs/libwb

if [[ ! -f "$LIBWB_TARGET/sources.cmake" ]]; then
	if [[ -f "$LIBWB_SOURCE/sources.cmake" ]]; then
		ln -sfn "$LIBWB_SOURCE" "$LIBWB_TARGET"
	else
		echo "ERROR: libwb not found. Expected one of:"
		echo "  $LIBWB_TARGET/sources.cmake"
		echo "  $LIBWB_SOURCE/sources.cmake"
		echo "Copy/provision libwb first, then rerun this script."
		exit 1
	fi
fi

rm -f CMakeCache.txt Makefile cmake_install.cmake compile_commands.json
rm -rf CMakeFiles

cmake -D CUDA_TOOLKIT_ROOT_DIR=/opt/ohpc/pub/apps/cuda/11.8 "$ASSIGNMENT3_DIR"

echo "Configuration complete. Next step: run 'make -j' on HPC when ready."