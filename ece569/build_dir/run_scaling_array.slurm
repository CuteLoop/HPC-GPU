#!/bin/bash

### Job name — %a is replaced with array task ID
#SBATCH --job-name=hw4_scale_%a

### Specify the group for this job
#SBATCH --account=ece569

### Set the queue for this job
#SBATCH --partition=gpu_standard

#SBATCH -e run_scaling_%a.error
#SBATCH -o run_scaling_%a.out

### Resources per task (one GPU per version)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb

### 30 min is plenty for 1 version × 8 sizes × 5 runs
#SBATCH --time=00:30:00

### Array: V0,V1,V2,V3,V4,V6,V7  (V5 excluded — O(N×B) would be hours at 50M)
#SBATCH --array=0-4,6-7

### Load required modules
module load cuda11/11.0
module load cmake/3.28.3

hw_path=/home/u16/joelmaldonado/HPC-GPU/ece569/build_dir
cd $hw_path

V=$SLURM_ARRAY_TASK_ID
SENTINEL="Histogram_output/scaling/.datasets_ready"

echo "========================================================="
echo " ECE 569 — HW4 Scaling  |  Version V${V}  |  $(date)"
echo "========================================================="

mkdir -p Histogram_output/scaling

# ── Dataset generation (task 0 only, all others wait) ────────────────────────
if [ "$V" -eq 0 ]; then
    echo "[V0] Generating datasets..."
    rm -f "$SENTINEL"
    ./Histogram_DatasetGenerator
    touch "$SENTINEL"
    echo "[V0] Datasets ready — sentinel written."
fi

echo "[V${V}] Waiting for datasets sentinel..."
WAITED=0
while [ ! -f "$SENTINEL" ]; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [ "$WAITED" -gt 300 ]; then
        echo "[V${V}] ERROR: timed out waiting for datasets after ${WAITED}s"
        exit 1
    fi
done
echo "[V${V}] Datasets confirmed. Starting benchmarks."

# ── Benchmark loop ────────────────────────────────────────────────────────────
# Dataset IDs → sizes: 11=2k, 12=10k, 13=50k, 14=200k, 6=500k, 8=1M, 9=10M, 10=50M
DATASETS=(11 12 13 14 6 8 9 10)
N_RUNS=5

for DS in "${DATASETS[@]}"; do
    echo "  V${V}  Dataset ${DS}"
    for RUN in $(seq 1 $N_RUNS); do
        OUTPUT_FILE="Histogram_output/scaling/v${V}_ds${DS}_run${RUN}.txt"
        /usr/bin/time -v ./Histogram_Solution \
            -e Histogram/Dataset/${DS}/output.raw \
            -i Histogram/Dataset/${DS}/input.raw \
            ${V} -t integral_vector > "${OUTPUT_FILE}" 2>&1
    done
done

echo "[V${V}] Done. $(date)"
