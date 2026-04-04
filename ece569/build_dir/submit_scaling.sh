#!/bin/bash
# submit_scaling.sh — submit the scaling array job + auto-cleanup
# Run this from: /home/u16/joelmaldonado/HPC-GPU/ece569/build_dir
#
# Usage:
#   bash submit_scaling.sh

hw_path=/home/u16/joelmaldonado/HPC-GPU/ece569/build_dir
cd $hw_path

# Clear stale sentinel so V0 always regenerates datasets fresh
rm -f Histogram_output/scaling/.datasets_ready

echo "Submitting scaling array job (V0,V1,V2,V3,V4,V6,V7)..."
JID=$(sbatch --parsable run_scaling_array.slurm)
echo "  Array job ID: ${JID}"

# Submit cleanup job that runs only after ALL array tasks succeed
CID=$(sbatch --parsable \
    --dependency=afterok:${JID} \
    --job-name=hw4_scale_cleanup \
    --account=ece569 \
    --partition=gpu_standard \
    --nodes=1 --ntasks=1 --mem=1gb --time=00:05:00 \
    --wrap="cd ${hw_path} && \
            rm -rf Histogram/Dataset/8 Histogram/Dataset/9 Histogram/Dataset/10 \
                   Histogram/Dataset/11 Histogram/Dataset/12 Histogram/Dataset/13 \
                   Histogram/Dataset/14 && \
            echo 'Dataset cleanup complete. Datasets 0-7 preserved.'")
echo "  Cleanup job ID: ${CID} (runs after all array tasks finish)"
echo ""
echo "Monitor: squeue -u \$(whoami)"
echo "Per-version logs: run_scaling_0.out ... run_scaling_7.out"
echo "Results will be in: Histogram_output/scaling/"
