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
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb

### Specify "wallclock time" - increase for perf sweep
#SBATCH --time 03:00:00
### Load required modules
module load cuda11/11.0
module load cmake/3.28.3

### set directory for job execution
hw_path=/home/u16/joelmaldonado/HPC-GPU/ece569/build_dir
cd $hw_path

########################################################################
# PART 1: 10 public test cases (for submission)
########################################################################

mkdir -p BasicMatrixMultiplication_output
mkdir -p TiledMatrixMultiplication_output

### Basic Matrix Multiplication - 10 public test cases
PATHS[0]=$hw_path/MatrixMultiplication/Dataset/0
PATHS[1]=$hw_path/MatrixMultiplication/Dataset/1
PATHS[2]=$hw_path/MatrixMultiplication/Dataset/2
PATHS[3]=$hw_path/MatrixMultiplication/Dataset/3
PATHS[4]=$hw_path/MatrixMultiplication/Dataset/4
PATHS[5]=$hw_path/MatrixMultiplication/Dataset/5
PATHS[6]=$hw_path/MatrixMultiplication/Dataset/6
PATHS[7]=$hw_path/MatrixMultiplication/Dataset/7
PATHS[8]=$hw_path/MatrixMultiplication/Dataset/8
PATHS[9]=$hw_path/MatrixMultiplication/Dataset/9

count=0
for i in ${PATHS[@]}
do
  file=output$((count)).txt
  ./BasicMatrixMultiplication_Solution -e $i/output.raw -i $i/input0.raw,$i/input1.raw -t matrix > BasicMatrixMultiplication_output/$file
  count=$((count+1))
done

### Tiled Matrix Multiplication - 10 public test cases
PATHS[0]=$hw_path/TiledMatrixMultiplication/Dataset/0
PATHS[1]=$hw_path/TiledMatrixMultiplication/Dataset/1
PATHS[2]=$hw_path/TiledMatrixMultiplication/Dataset/2
PATHS[3]=$hw_path/TiledMatrixMultiplication/Dataset/3
PATHS[4]=$hw_path/TiledMatrixMultiplication/Dataset/4
PATHS[5]=$hw_path/TiledMatrixMultiplication/Dataset/5
PATHS[6]=$hw_path/TiledMatrixMultiplication/Dataset/6
PATHS[7]=$hw_path/TiledMatrixMultiplication/Dataset/7
PATHS[8]=$hw_path/TiledMatrixMultiplication/Dataset/8
PATHS[9]=$hw_path/TiledMatrixMultiplication/Dataset/9

count=0
for j in ${PATHS[@]}
do
  file=output$((count)).txt
  ./TiledMatrixMultiplication_Solution -e $j/output.raw -i $j/input0.raw,$j/input1.raw -t matrix > TiledMatrixMultiplication_output/$file
  count=$((count+1))
done

########################################################################
# PART 2: Performance sweep (Task 3)
# Uses the sweep executables that iterate block sizes 4,8,16,32
# and use cudaEvent timing for kernel-only measurement.
# Each case is run 10 times for averaging.
########################################################################

NUM_RUNS=10
PERF_DIR=$hw_path/perf_sweep_output
mkdir -p $PERF_DIR

### Performance test case datasets (generated as datasets 10-13)
### Case 0: 2048x2048 * 2048x2048
### Case 1: 2051x2051 * 2051x2051
### Case 2: 4096x2048 * 2048x4096
### Case 3: 4096x2047 * 2047x4095

BASIC_PERF_PATHS[0]=$hw_path/MatrixMultiplication/Dataset/10
BASIC_PERF_PATHS[1]=$hw_path/MatrixMultiplication/Dataset/11
BASIC_PERF_PATHS[2]=$hw_path/MatrixMultiplication/Dataset/12
BASIC_PERF_PATHS[3]=$hw_path/MatrixMultiplication/Dataset/13

TILED_PERF_PATHS[0]=$hw_path/TiledMatrixMultiplication/Dataset/10
TILED_PERF_PATHS[1]=$hw_path/TiledMatrixMultiplication/Dataset/11
TILED_PERF_PATHS[2]=$hw_path/TiledMatrixMultiplication/Dataset/12
TILED_PERF_PATHS[3]=$hw_path/TiledMatrixMultiplication/Dataset/13

### Basic sweep: 4 cases x 10 runs (each run tests all 4 block sizes)
echo "===== BASIC PERFORMANCE SWEEP ====="
for case_idx in 0 1 2 3
do
  echo "Basic Case $case_idx:"
  dp=${BASIC_PERF_PATHS[$case_idx]}
  for run in $(seq 1 $NUM_RUNS)
  do
    ./basic_sweep -e $dp/output.raw -i $dp/input0.raw,$dp/input1.raw -t matrix >> $PERF_DIR/basic_case${case_idx}.txt 2>&1
  done
  echo "  Done ($NUM_RUNS runs)"
done

### Tiled sweep: 4 cases x 10 runs (each run tests all 4 block sizes)
echo "===== TILED PERFORMANCE SWEEP ====="
for case_idx in 0 1 2 3
do
  echo "Tiled Case $case_idx:"
  dp=${TILED_PERF_PATHS[$case_idx]}
  for run in $(seq 1 $NUM_RUNS)
  do
    ./tiled_sweep -e $dp/output.raw -i $dp/input0.raw,$dp/input1.raw -t matrix >> $PERF_DIR/tiled_case${case_idx}.txt 2>&1
  done
  echo "  Done ($NUM_RUNS runs)"
done

echo ""
echo "===== ALL DONE ====="
echo "Public test outputs in: BasicMatrixMultiplication_output/ and TiledMatrixMultiplication_output/"
echo "Performance sweep data in: $PERF_DIR/"
echo "Next step: python3 parse_and_plot.py $PERF_DIR"
