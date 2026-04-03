#!/bin/bash

### Set the job name 
#SBATCH --job-name=ece569hw4

### Specify the group for this job
### List of PI groups available to each user can be found with "va" command

####ece569 students can also replace ece569 with another group id if sponsored by another faculty. 
#SBATCH --account=ece569

### Set the queue for this job as windfall or standard (adjust ### and #)
#SBATCH --partition=gpu_standard

#SBATCH -e run_hw4.error
#SBATCH -o run_hw4.out

### Set the number of nodes, cores and memory that will be used for this job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb

### Specify "wallclock time", hhh:mm:ss. Required field
#SBATCH --time 00:05:00

### Load required modules
module load cuda11/11.0
module load cmake/3.28.3

### set directory for job execution
hw_path=/home/u16/joelmaldonado/HPC-GPU/ece569/build_dir

cd $hw_path

### Create output directory
mkdir -p Histogram_output

### Generate datasets
./Histogram_DatasetGenerator

###
PATHS[0]=$hw_path/Histogram/Dataset/0
PATHS[1]=$hw_path/Histogram/Dataset/1
PATHS[2]=$hw_path/Histogram/Dataset/2
PATHS[3]=$hw_path/Histogram/Dataset/3
PATHS[4]=$hw_path/Histogram/Dataset/4
PATHS[5]=$hw_path/Histogram/Dataset/5
PATHS[6]=$hw_path/Histogram/Dataset/6

### Version 0: global memory kernel
count=0
for j in ${PATHS[@]}
do
   file=output_v0_$((count)).txt
   /usr/bin/time ./Histogram_Solution -e $j/output.raw -i $j/input.raw 0 -t integral_vector > Histogram_output/$file
   count=$((count+1))
done

### Version 1: shared memory privatized kernel
count=0
for j in ${PATHS[@]}
do
   file=output_v1_$((count)).txt
   /usr/bin/time ./Histogram_Solution -e $j/output.raw -i $j/input.raw 1 -t integral_vector > Histogram_output/$file
   count=$((count+1))
done

### Version 2: THE GOAT — R=2 per-block + RLE + uint4
count=0
for j in ${PATHS[@]}
do
   file=output_v2_$((count)).txt
   /usr/bin/time ./Histogram_Solution -e $j/output.raw -i $j/input.raw 2 -t integral_vector > Histogram_output/$file
   count=$((count+1))
done

### Version 3: ablation — R=2 per-block + uint4, no RLE
count=0
for j in ${PATHS[@]}
do
   file=output_v3_$((count)).txt
   /usr/bin/time ./Histogram_Solution -e $j/output.raw -i $j/input.raw 3 -t integral_vector > Histogram_output/$file
   count=$((count+1))
done

### Version 4: warp aggregation (__ballot_sync/__popc)
count=0
for j in ${PATHS[@]}
do
   file=output_v4_$((count)).txt
   /usr/bin/time ./Histogram_Solution -e $j/output.raw -i $j/input.raw 4 -t integral_vector > Histogram_output/$file
   count=$((count+1))
done

### Version 5: bin-centric gather (zero atomics)
count=0
for j in ${PATHS[@]}
do
   file=output_v5_$((count)).txt
   /usr/bin/time ./Histogram_Solution -e $j/output.raw -i $j/input.raw 5 -t integral_vector > Histogram_output/$file
   count=$((count+1))
done

### Version 6: sort & reduce-by-key via Thrust (sorts d_input in-place; fresh load each run)
count=0
for j in ${PATHS[@]}
do
   file=output_v6_$((count)).txt
   /usr/bin/time ./Histogram_Solution -e $j/output.raw -i $j/input.raw 6 -t integral_vector > Histogram_output/$file
   count=$((count+1))
done

### Version 7: multi-split hierarchical bucketing
count=0
for j in ${PATHS[@]}
do
   file=output_v7_$((count)).txt
   /usr/bin/time ./Histogram_Solution -e $j/output.raw -i $j/input.raw 7 -t integral_vector > Histogram_output/$file
   count=$((count+1))
done

echo "===== HW4 DONE ====="
echo "Check outputs in Histogram_output/"



