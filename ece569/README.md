# ECE 569 — HPC/GPU Homework Guide

## Repository Layout

```
HPC-GPU/
├── Assignment2/          # Instructor-provided HW2 templates (reference only)
├── Assignment3/          # Instructor-provided HW3 templates (reference only)
├── Assignment4/          # Instructor-provided HW4 templates (reference only)
└── ece569/
    ├── labs/             # YOUR SOURCE CODE lives here
    │   ├── libwb/        # Shared helper library (do not modify)
    │   ├── hw2/
    │   │   ├── VectorAdd/
    │   │   │   └── solution.cu          ← your hw2 vector add code
    │   │   └── ImageColorToGrayscale/
    │   │       └── solution.cu          ← your hw2 grayscale code
    │   ├── hw3/
    │   │   ├── BasicMatrixMultiplication/
    │   │   │   └── solution.cu          ← your hw3 basic matmul code
    │   │   ├── TiledMatrixMultiplication/
    │   │   │   └── solution.cu          ← your hw3 tiled matmul code
    │   │   └── performance/
    │   │       ├── basic_solution.cu    ← hw3 perf sweep (basic)
    │   │       └── tiled_solution.cu    ← hw3 perf sweep (tiled)
    │   └── hw4/
    │       └── Histogram/
    │           ├── kernel.cu            ← YOUR hw4 kernel code (edit this)
    │           └── solution.cu          ← harness (do not modify)
    └── build_dir/        # ALL build outputs go here (cmake out-of-source build)
        ├── configure_hw2.sh
        ├── configure_hw3.sh
        ├── configure_hw4.sh
        ├── run_hw2.slurm
        ├── run_hw3.slurm
        └── (binaries, datasets, output folders appear here after build)
```

---

## Workflow (every homework)

### Step 1 — Edit your solution file on HPC

All your code goes inside `ece569/labs/hwX/LabName/solution.cu`  
For HW4, edit `ece569/labs/hw4/Histogram/kernel.cu` (the harness is in `solution.cu`).

### Step 2 — Configure (only needed once per new HW or after wiping build_dir)

SSH into the HPC, then:

```bash
cd ~/HPC-GPU/ece569/build_dir
bash configure_hw4.sh     # or configure_hw2.sh / configure_hw3.sh
```

All configure scripts are identical — they all point cmake at `ece569/labs/`  
and build into `ece569/build_dir/`. No symlinks needed.

### Step 3 — Build

```bash
cd ~/HPC-GPU/ece569/build_dir
make -j
```

Successful build produces binaries directly in `build_dir/`, e.g.:
- `VectorAdd_Solution`
- `ImageColorToGrayscale_Solution`
- `BasicMatrixMultiplication_Solution`
- `TiledMatrixMultiplication_Solution`
- `Histogram_Solution`
- `*_DatasetGenerator` variants

### Step 4 — Run via SLURM

```bash
cd ~/HPC-GPU/ece569/build_dir
sbatch run_hw2.slurm    # submits HW2 test runs
sbatch run_hw3.slurm    # submits HW3 test runs
```

Check job status:
```bash
squeue --me
```

Check output after job finishes:
```bash
cat run.out
cat run.error
```

---

## Per-Homework Details

### HW2 — Vector Addition & Image Grayscale

**Source files to edit:**
- `ece569/labs/hw2/VectorAdd/solution.cu`
- `ece569/labs/hw2/ImageColorToGrayscale/solution.cu`

**Binaries produced:**
- `VectorAdd_Solution` — takes 2 float vector inputs, produces summed output
- `ImageColorToGrayscale_Solution` — takes a `.ppm` color image, produces `.pbm` grayscale

**Manual run example (from build_dir):**
```bash
./VectorAdd_Solution \
  -e VectorAdd/Dataset/0/output.raw \
  -i VectorAdd/Dataset/0/input0.raw,VectorAdd/Dataset/0/input1.raw \
  -t vector

./ImageColorToGrayscale_Solution \
  -e ImageColorToGrayscale/Dataset/0/output.pbm \
  -i ImageColorToGrayscale/Dataset/0/input.ppm \
  -t image
```

**SLURM script:** `build_dir/run_hw2.slurm` — runs all 10 dataset cases for each lab, saves results to `ImageColorToGrayScale_output/` and `VectorAdd_output/`.

---

### HW3 — Matrix Multiplication

**Source files to edit:**
- `ece569/labs/hw3/BasicMatrixMultiplication/solution.cu`
- `ece569/labs/hw3/TiledMatrixMultiplication/solution.cu`
- `ece569/labs/hw3/performance/basic_solution.cu` (performance sweep variant)
- `ece569/labs/hw3/performance/tiled_solution.cu` (performance sweep variant)

**Binaries produced:**
- `BasicMatrixMultiplication_Solution`
- `TiledMatrixMultiplication_Solution`
- `BasicPerformance_Solution`
- `TiledPerformance_Solution`

**Manual run example (from build_dir):**
```bash
./BasicMatrixMultiplication_Solution \
  -e MatrixMultiplication/Dataset/0/output.raw \
  -i MatrixMultiplication/Dataset/0/input0.raw,MatrixMultiplication/Dataset/0/input1.raw \
  -t matrix

./TiledMatrixMultiplication_Solution \
  -e TiledMatrixMultiplication/Dataset/0/output.raw \
  -i TiledMatrixMultiplication/Dataset/0/input0.raw,TiledMatrixMultiplication/Dataset/0/input1.raw \
  -t matrix
```

**SLURM script:** `build_dir/run_hw3.slurm` — runs 10 dataset cases for each kernel, saves to `BasicMatrixMultiplication_output/` and `TiledMatrixMultiplication_output/`.

---

### HW4 — Histogram

**Source file to edit:** `ece569/labs/hw4/Histogram/kernel.cu`  
**Do NOT edit:** `ece569/labs/hw4/Histogram/solution.cu` (it is the test harness)

**Kernels to implement in `kernel.cu`:**

| Kernel | Version flag | Description |
|---|---|---|
| `histogram_global_kernel` | `0` | Global memory atomics only |
| `histogram_shared_kernel` | `1` | Shared memory privatization |
| `histogram_shared_optimized` | `2` | Your best optimization |
| `convert_kernel` | called by all | Clips bin values > 127 to 127 |

**Binary produced:** `Histogram_Solution`

**Manual run example (from build_dir):**
```bash
# Version 0 (global memory)
./Histogram_Solution \
  -e Histogram/Dataset/0/output.raw \
  -i Histogram/Dataset/0/input.raw \
  -o /tmp/hist_out.raw \
  -t integral_vector 0

# Version 1 (shared memory)
./Histogram_Solution \
  -e Histogram/Dataset/0/output.raw \
  -i Histogram/Dataset/0/input.raw \
  -o /tmp/hist_out.raw \
  -t integral_vector 1

# Version 2 (optimized)
./Histogram_Solution \
  -e Histogram/Dataset/0/output.raw \
  -i Histogram/Dataset/0/input.raw \
  -o /tmp/hist_out.raw \
  -t integral_vector 2
```

**Grid configuration (fixed in harness):** `<<<30 blocks, 512 threads>>>`, `NUM_BINS = 4096`

**SLURM script:** `Assignment4/Assignment4/run_hw4.slurm` — update `hw_path` to point to your `build_dir` before submitting. Change the last argument (`0`, `1`, `2`) to test each version.

---

## Regenerating Datasets

Each lab has a `*_DatasetGenerator` binary. Run it to regenerate test datasets if they're missing:

```bash
./VectorAdd_DatasetGenerator
./ImageColorToGrayscale_DatasetGenerator
./BasicMatrixMultiplication_DatasetGenerator
./TiledMatrixMultiplication_DatasetGenerator
./Histogram_DatasetGenerator
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `cmake` fails to find CUDA | Make sure `module load cuda11/11.0` ran before configure |
| `make` finds no targets | Re-run the configure script to clear stale `CMakeCache.txt` |
| Binary not found after `make` | Check `build_dir/` — binaries land here, not in `labs/` |
| SLURM job fails immediately | Check `run.error` in `build_dir/`; also verify `hw_path` in the slurm script matches your actual HPC path |
| `wbCheck` failure in output | Your kernel has a bug — check array bounds, shared memory size, or sync points |
