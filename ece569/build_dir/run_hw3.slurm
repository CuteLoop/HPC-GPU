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

echo "===== PART 1 DONE ====="
echo "Check outputs in BasicMatrixMultiplication_output/ and TiledMatrixMultiplication_output/"
