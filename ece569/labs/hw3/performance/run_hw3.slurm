#!/bin/bash
### Set the job name
#SBATCH --job-name=ece569hw3
### Specify the group for this job
#SBATCH --account=ece569
### Set the queue for this job
#SBATCH --partition=gpu_standard
#SBATCH -e run_hw3.error
#SBATCH -o run_hw3.out
### Set the number of nodes, cores and memory that will be used for this job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-atask=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb
### Specify "wallclock time"
#SBATCH --time 00:05:00
### Load required modules
module load cuda11/11.0
module load cmake/3.28.3

### set directory for job execution
hw_path=/home/u16/joelmaldonado/HPC-GPU/ece569/build_dir
cd $hw_path

###
mkdir -p BasicMatrixMultiplication_output
mkdir -p TiledMatrixMultiplication_output

# Performance sweep settings: run only generated datasets and repeat each case.
REPEATS=10
DATASET_IDS=(10 11 12 13)

### Basic Matrix Multiplication - generated test cases only (10 to 13)
for dataset_id in ${DATASET_IDS[@]}
do
  path=$hw_path/MatrixMultiplication/Dataset/$dataset_id
  for run in $(seq 1 $REPEATS)
  do
    file=output${dataset_id}_run${run}.txt
    ./BasicMatrixMultiplication_Solution -e $path/output.raw -i $path/input0.raw,$path/input1.raw -t matrix > BasicMatrixMultiplication_output/$file
  done
done

### Tiled Matrix Multiplication - generated test cases only (10 to 13)
for dataset_id in ${DATASET_IDS[@]}
do
  path=$hw_path/TiledMatrixMultiplication/Dataset/$dataset_id
  for run in $(seq 1 $REPEATS)
  do
    file=output${dataset_id}_run${run}.txt
    ./TiledMatrixMultiplication_Solution -e $path/output.raw -i $path/input0.raw,$path/input1.raw -t matrix > TiledMatrixMultiplication_output/$file
  done
done

echo "===== PART 1 DONE ====="
echo "Check outputs in BasicMatrixMultiplication_output/ and TiledMatrixMultiplication_output/"
