#!/bin/bash
#SBATCH --job-name=hw4_exp1
#SBATCH --account=ece569
#SBATCH --partition=gpu_standard
#SBATCH -e exp1.error
#SBATCH -o exp1.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb
#SBATCH --time 00:10:00

module load cuda11/11.0
module load cmake/3.28.3

hw_path=/home/u16/joelmaldonado/HPC-GPU/ece569/build_dir
DATASET=$hw_path/Histogram/Dataset/6   # dataset 6: 500k random elements

cd $hw_path
mkdir -p Histogram_output/exp1 Histogram_output/figures

# Ensure datasets exist (DatasetGenerator is idempotent)
if [ ! -f "$DATASET/input.raw" ]; then
    echo "Dataset 6 not found — running DatasetGenerator..."
    ./Histogram_DatasetGenerator
fi

echo "===== Experiment 1: Dataset 6 (random, 500k elements) ====="
echo "10 runs per version — reading kernel time from each output file"

# --- Version 0 ---
echo ""
echo "--- Version 0: global memory interleaved ---"
for run in $(seq 1 10); do
    outfile=Histogram_output/exp1/v0_run${run}.txt
    ./Histogram_Solution -e $DATASET/output.raw -i $DATASET/input.raw 0 -t integral_vector > $outfile 2>&1
    echo "  run $run: $(grep 'Total compute time' $outfile)"
done

# --- Version 1 ---
echo ""
echo "--- Version 1: shared memory privatized ---"
for run in $(seq 1 10); do
    outfile=Histogram_output/exp1/v1_run${run}.txt
    ./Histogram_Solution -e $DATASET/output.raw -i $DATASET/input.raw 1 -t integral_vector > $outfile 2>&1
    echo "  run $run: $(grep 'Total compute time' $outfile)"
done

# --- Version 2 ---
echo ""
echo "--- Version 2: THE GOAT (R=2 per-block + RLE + uint4) ---"
for run in $(seq 1 10); do
    outfile=Histogram_output/exp1/v2_run${run}.txt
    ./Histogram_Solution -e $DATASET/output.raw -i $DATASET/input.raw 2 -t integral_vector > $outfile 2>&1
    echo "  run $run: $(grep 'Total compute time' $outfile)"
done

# --- Version 3 ---
echo ""
echo "--- Version 3: ablation (R=2 + uint4, no RLE) ---"
for run in $(seq 1 10); do
    outfile=Histogram_output/exp1/v3_run${run}.txt
    ./Histogram_Solution -e $DATASET/output.raw -i $DATASET/input.raw 3 -t integral_vector > $outfile 2>&1
    echo "  run $run: $(grep 'Total compute time' $outfile)"
done

# --- Version 4 ---
echo ""
echo "--- Version 4: warp aggregation (__ballot_sync/__popc) ---"
for run in $(seq 1 10); do
    outfile=Histogram_output/exp1/v4_run${run}.txt
    ./Histogram_Solution -e $DATASET/output.raw -i $DATASET/input.raw 4 -t integral_vector > $outfile 2>&1
    echo "  run $run: $(grep 'Total compute time' $outfile)"
done

# --- Version 5 ---
echo ""
echo "--- Version 5: bin-centric gather (zero atomics) ---"
for run in $(seq 1 10); do
    outfile=Histogram_output/exp1/v5_run${run}.txt
    ./Histogram_Solution -e $DATASET/output.raw -i $DATASET/input.raw 5 -t integral_vector > $outfile 2>&1
    echo "  run $run: $(grep 'Total compute time' $outfile)"
done

# --- Version 6 ---
echo ""
echo "--- Version 6: sort & reduce-by-key (Thrust) ---"
for run in $(seq 1 10); do
    outfile=Histogram_output/exp1/v6_run${run}.txt
    ./Histogram_Solution -e $DATASET/output.raw -i $DATASET/input.raw 6 -t integral_vector > $outfile 2>&1
    echo "  run $run: $(grep 'Total compute time' $outfile)"
done

# --- Version 7 ---
echo ""
echo "--- Version 7: multi-split hierarchical bucketing ---"
for run in $(seq 1 10); do
    outfile=Histogram_output/exp1/v7_run${run}.txt
    ./Histogram_Solution -e $DATASET/output.raw -i $DATASET/input.raw 7 -t integral_vector > $outfile 2>&1
    echo "  run $run: $(grep 'Total compute time' $outfile)"
done

echo ""
echo "===== Experiment 1 DONE ====="
echo "Outputs in Histogram_output/exp1/"
