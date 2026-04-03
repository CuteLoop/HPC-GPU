#!/bin/bash

### Set the job name
#SBATCH --job-name=hw4_scaling

### Specify the group for this job
#SBATCH --account=ece569

### Set the queue for this job
#SBATCH --partition=gpu_standard

#SBATCH -e run_scaling.error
#SBATCH -o run_scaling.out

### Set the number of nodes, cores and memory that will be used for this job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb

### Specify "wallclock time" — all 8 versions x 8 dataset sizes x 5 runs
#SBATCH --time 01:00:00

### Load required modules
module load cuda11/11.0
module load cmake/3.28.3

### Set directory for job execution
hw_path=/home/u16/joelmaldonado/HPC-GPU/ece569/build_dir
cd $hw_path

echo "========================================================="
echo " ECE 569 - HW4 Asymptotic Scaling Profiling"
echo "========================================================="

### Create output directory for scaling results
mkdir -p Histogram_output/scaling

echo "--- Step 1: Generating datasets (~1 GB, temporary) ---"
./Histogram_DatasetGenerator

# Dataset IDs → sizes: 11=2k, 12=10k, 13=50k, 14=200k, 6=500k, 8=1M, 9=10M, 10=50M
DATASETS=(11 12 13 14 6 8 9 10)
VERSIONS=(0 1 2 3 4 5 6 7)   # All histogram versions
N_RUNS=5

echo "--- Step 2: Running benchmarks ---"
for V in "${VERSIONS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        echo "  V${V}  Dataset ${DS}"
        for RUN in $(seq 1 $N_RUNS); do
            OUTPUT_FILE="Histogram_output/scaling/v${V}_ds${DS}_run${RUN}.txt"
            /usr/bin/time -v ./Histogram_Solution \
                -e Histogram/Dataset/${DS}/output.raw \
                -i Histogram/Dataset/${DS}/input.raw \
                ${V} -t integral_vector > ${OUTPUT_FILE} 2>&1
        done
    done
done

echo "--- Step 3: Deleting large dataset files to recover quota ---"
rm -rf Histogram/Dataset/8
rm -rf Histogram/Dataset/9
rm -rf Histogram/Dataset/10
rm -rf Histogram/Dataset/11
rm -rf Histogram/Dataset/12
rm -rf Histogram/Dataset/13
rm -rf Histogram/Dataset/14

echo "Cleanup complete."
echo "Timing results saved to: Histogram_output/scaling/"
echo "Datasets 0-6 preserved for standard hw4 experiments."
