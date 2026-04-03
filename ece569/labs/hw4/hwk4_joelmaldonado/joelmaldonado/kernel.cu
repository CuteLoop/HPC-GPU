// version 0 — Global Scatter
// Core Pattern: each thread scatters its element directly into the global bin.
// Atomic Scope: global — every increment fires a global atomicAdd (N total per GPU block).
// Memory Access: 32-bit coalesced loads; each warp reads 32 consecutive unsigned ints in one
// 128-byte transaction (fully coalesced on P100), but every atomicAdd hits DRAM-backed L2.
// Exp 1 (Random): Slow — N global atomics, heavy contention on hot bins in L2/DRAM.
// Exp 2 (Uniform): Catastrophic — all N threads hammer the same bin; serialized by hardware.
__global__ void histogram_global_kernel(unsigned int *input,
     unsigned int *bins, unsigned int num_elements, unsigned int num_bins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (i < num_elements) {
        unsigned int number = input[i];
        if (number < num_bins) {
            atomicAdd(&bins[number], 1);
        }
        i += stride;
    }

}



// version 1 — Block Privatization
// Core Pattern: each block maintains a private histogram in shared memory; merges at end.
// Atomic Scope: shared — N atomicAdds per block land in L1 shared memory (fast, ~4 cycles).
// After all threads finish, one merge pass fires num_bins global atomicAdds per block.
// Memory Access: 32-bit coalesced loads (same as V0).
// Exp 1 (Random): Fast — shared atomics orders of magnitude cheaper than global.
// Exp 2 (Uniform): Extremely Slow — all threads within a block still serialize on the
//   same single shared-memory bin; N/blockDim serialized increments per block.
__global__ void histogram_shared_kernel(unsigned int *input, unsigned int *bins,
     unsigned int num_elements, unsigned int num_bins) {
        extern __shared__ unsigned int histogram_private[];
        // initialize all bins in shared memory - loop covers all num_bins even if > blockDim.x
        for (int b = threadIdx.x; b < num_bins; b += blockDim.x)
            histogram_private[b] = 0;
        __syncthreads();
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int stride = blockDim.x * gridDim.x;
        while (i < num_elements) {
            unsigned int number = input[i];
            if (number < num_bins) {
                atomicAdd(&histogram_private[number], 1);
            }
            i += stride;
        }
        // wait for all threads to finish updating the private histogram
        __syncthreads();
        // merge private histogram into global - loop covers all num_bins
        for (int b = threadIdx.x; b < num_bins; b += blockDim.x) {
            atomicAdd(&bins[b], histogram_private[b]);
        }
     }
// version 2 — Warp Aggregation (Register-Level RLE + Vectorized 128-bit Loads)
// Core Pattern: register-level run-length encoding acts as implicit warp aggregation.
//   Each thread accumulates consecutive same-bin elements into a register counter;
//   an atomicAdd to shared memory fires only on a bin change or end of work.
//   For a random input with W distinct bins touched per warp, at most ~N/32 atomics fire.
// Atomic Scope: shared — Max ~N/32 per warp (compared to N for V1). For uniform data
//   every thread fires exactly 1 atomicAdd regardless of N: O(1) per thread.
// Memory Access: 128-bit uint4 loads — each LDG.E.128 instruction fetches 4 unsigned ints,
//   quadrupling effective bandwidth over V0/V1's 32-bit scalar loads on P100.
// Exp 1 (Random): Fastest — 4x BW gain + fewer shared atomics than V1.
// Exp 2 (Uniform): Fastest — single atomicAdd per thread (complete RLE compression).
// Source: CUDA C++ Programming Guide vectorized loads; Kirk & Hwu PMPP histogram optimization.
__global__ void histogram_shared_optimized(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
    extern __shared__ unsigned int histogram_private[];

    // 1. Initialize shared memory bins to 0
    for (int b = threadIdx.x; b < num_bins; b += blockDim.x) {
        histogram_private[b] = 0;
    }
    __syncthreads();

    // 2. Vectorized load setup: reinterpret as 128-bit (uint4) array
    uint4 *vector_input = reinterpret_cast<uint4*>(input);
    int num_vectors = num_elements / 4;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 3. RLE state in registers
    unsigned int current_bin = 0xFFFFFFFF; // sentinel: no active run
    unsigned int run_count = 0;

    // 4. Main loop: 128-bit vectorized loads + RLE compression
    while (i < num_vectors) {
        uint4 data = vector_input[i];
        unsigned int vals[4] = {data.x, data.y, data.z, data.w};

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            unsigned int val = vals[k];
            if (val < num_bins) {
                if (val == current_bin) {
                    run_count++; // same bin: pure register increment, no atomic
                } else {
                    if (run_count > 0) {
                        atomicAdd(&histogram_private[current_bin], run_count);
                    }
                    current_bin = val;
                    run_count = 1;
                }
            }
        }
        i += stride;
    }

    // 5. Flush final RLE run from registers to shared memory
    if (run_count > 0 && current_bin < num_bins) {
        atomicAdd(&histogram_private[current_bin], run_count);
    }

    // 6. Tail: handle remaining elements if num_elements % 4 != 0
    int tail_start = num_vectors * 4;
    int tail_i = tail_start + blockIdx.x * blockDim.x + threadIdx.x;
    while (tail_i < num_elements) {
        unsigned int val = input[tail_i];
        if (val < num_bins) {
            atomicAdd(&histogram_private[val], 1);
        }
        tail_i += stride;
    }

    // 7. Merge private histogram into global bins
    __syncthreads();
    for (int b = threadIdx.x; b < num_bins; b += blockDim.x) {
        if (histogram_private[b] > 0) {
            atomicAdd(&bins[b], histogram_private[b]);
        }
    }
}

// version 3 — Bin-Centric Gather
// Core Pattern: inverts scatter (thread→bin) to gather (bin→thread).
//   Each thread owns exactly one bin and scans the entire input looking for its value.
// Atomic Scope: None (Zero) — each bin is written by exactly one thread; no atomics anywhere.
// Memory Access: Collaborative Tiling — a block of 256 threads loads a 256-element tile from
//   global memory in a single coalesced transaction, then all threads scan the tile in registers.
//   Trades atomic hardware contention for extra arithmetic (O(N*num_bins) comparisons total).
// Clipping: applied directly inside kernel (no convert_kernel pass needed).
// Exp 1 (Random): Moderate — zero atomics but O(N) scan per bin = O(N*B) total work.
// Exp 2 (Uniform): Very Fast — still O(N*B) arithmetic, but zero atomic serialization;
//   every bin sees identical latency regardless of distribution skew.
__global__ void histogram_gather_kernel(unsigned int *input, unsigned int *bins,
                                        unsigned int num_elements, unsigned int num_bins) {

    // Thread-to-bin mapping: each thread owns one bin
    unsigned int my_bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (my_bin >= num_bins) return;

    unsigned int local_count = 0;

    // Shared memory tile: collaboratively loaded by the block
    extern __shared__ unsigned int tile[];

    // Stream through entire input in blockDim.x-sized chunks
    for (unsigned int i = 0; i < num_elements; i += blockDim.x) {

        // Collaborative coalesced load of one tile from global memory
        if (i + threadIdx.x < num_elements) {
            tile[threadIdx.x] = input[i + threadIdx.x];
        }
        __syncthreads(); // wait for full tile

        int valid_elements = min((int)blockDim.x, (int)(num_elements - i));

        // Each thread scans the tile for its own bin — no atomics
        #pragma unroll 4
        for (int j = 0; j < valid_elements; j++) {
            if (tile[j] == my_bin) {
                local_count++;
            }
        }
        __syncthreads(); // wait before overwriting tile
    }

    // One non-atomic global write per thread — each bin written by exactly one thread
    // Apply clipping: cap at 127 per spec
    bins[my_bin] = min(local_count, 127u);
}

// version 4 — Sequential (single-thread GPU baseline)
// Core Pattern: direct port of the CPU sequential algorithm to a single GPU thread.
//   One thread processes all N elements in order: reads input[i], increments bins[val].
//   Semantically identical to the CPU sequential_histogram reference implementation,
//   but executed on the GPU to isolate GPU overhead vs. parallelism.
// Atomic Scope: None — single thread owns all bins; no concurrent writes possible.
// Memory Access: 32-bit scalar sequential (no coalescing, no vectorization).
// Exp 1 (Random): Slowest — no parallelism; O(N) sequential global memory accesses.
// Exp 2 (Uniform): Slow — same O(N) sequential cost regardless of distribution.
// Based on: sequential_histogram.cpp (attached reference implementation).
__global__ void histogram_sequential_kernel(unsigned int *input, unsigned int *bins,
                                            unsigned int num_elements,
                                            unsigned int num_bins) {
    // Guard: only thread 0 in block 0 does any work — true sequential execution
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // Initialize bins to 0 (mirrors sequential_histogram's implicit zero-init)
    for (unsigned int b = 0; b < num_bins; b++) {
        bins[b] = 0;
    }

    // Sequential scan — identical logic to sequential_histogram.cpp, adapted for
    // unsigned int input instead of char (bin index = input value directly)
    for (unsigned int i = 0; i < num_elements; i++) {
        unsigned int val = input[i];
        if (val < num_bins) {
            bins[val]++;   // no atomics needed — single thread owns all bins
        }
    }

    // Apply clipping: cap each bin at 127 per spec (mirrors convert_kernel)
    for (unsigned int b = 0; b < num_bins; b++) {
        if (bins[b] > 127) bins[b] = 127;
    }
}

// clipping function
// resets bins that have value larger than 127 to 127.
// that is if bin[i]>127 then bin[i]=127

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

    // clipping. if number is larger than 127, set it to 127
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_bins) {
        if (bins[idx] > 127) {
            bins[idx] = 127;
        }
    }
}
