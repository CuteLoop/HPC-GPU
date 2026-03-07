#!/bin/bash
### Set the job name
#SBATCH --job-name=ece569hw3perf
### Specify the group for this job
#SBATCH --account=ece569
### Set the queue for this job
#SBATCH --partition=gpu_standard
#SBATCH -e run_hw3_perf.error
#SBATCH -o run_hw3_perf.out
### Set the number of nodes, cores and memory that will be used for this job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb
### Specify "wallclock time"
#SBATCH --time 00:10:00

### Load required modules
module load cuda11/11.0
module load cmake/3.28.3

### Run from build_dir (submit with: sbatch run_hw3_perf.slurm)
hw_path=$(pwd)
cd "$hw_path"

### Output and organization folders
mkdir -p BasicMatrixMultiplication_output
mkdir -p TiledMatrixMultiplication_output
mkdir -p BasicPerformance
mkdir -p BasicPerformance_output
mkdir -p TiledPerformance
mkdir -p TiledPerformance_output

### Generate performance datasets (cases 10-13)
./BasicPerformance_DatasetGenerator
./TiledPerformance_DatasetGenerator

# Performance sweep settings: run only generated datasets and repeat each case.
REPEATS=10
DATASET_IDS=(10 11 12 13)
BLOCK_SIZES=(4 8 16 32)

### Basic Matrix Multiplication - generated test cases only (10 to 13)
for block_size in ${BLOCK_SIZES[@]}
do
  export BLOCK_SIZE=$block_size
  for dataset_id in ${DATASET_IDS[@]}
  do
    path=$hw_path/MatrixMultiplication/Dataset/$dataset_id
    for run in $(seq 1 $REPEATS)
    do
      file=bs${block_size}_case${dataset_id}_run${run}.txt
      ./BasicPerformance_Solution -e $path/output.raw -i $path/input0.raw,$path/input1.raw -t matrix > BasicPerformance_output/$file
    done
  done
done

### Tiled Matrix Multiplication - generated test cases only (10 to 13)
for block_size in ${BLOCK_SIZES[@]}
do
  export BLOCK_SIZE=$block_size
  for dataset_id in ${DATASET_IDS[@]}
  do
    path=$hw_path/TiledMatrixMultiplication/Dataset/$dataset_id
    for run in $(seq 1 $REPEATS)
    do
      file=bs${block_size}_case${dataset_id}_run${run}.txt
      ./TiledPerformance_Solution -e $path/output.raw -i $path/input0.raw,$path/input1.raw -t matrix > TiledPerformance_output/$file
    done
  done
done

echo "===== PERFORMANCE RUN DONE ====="
echo "Check outputs in BasicPerformance_output/ and TiledPerformance_output/"
