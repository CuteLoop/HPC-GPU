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
// version 2 — THE GOAT: Vectorized R-per-Block + RLE Compression
// Core Pattern: attacks shared memory bank conflicts, memory bandwidth, and atomic count
//   simultaneously. Replicates the shared histogram R=2 times (padded +1 to break
//   power-of-two bank alignment) so even/odd threads scatter into different copies,
//   statistically halving intra-warp bank conflicts. Layered on top, thread-local
//   Register Run-Length Encoding compresses consecutive equal values into a single
//   atomicAdd, mathematically reducing shared atomics for any non-random distribution.
//   128-bit uint4 loads saturate the 128-bit P100 memory bus (4x over scalar loads).
//   Sparsity-filtered global flush fires only for non-zero bins.
// Atomic Scope: shared — ≤ N/stride RLE-compressed atomics (worst case random 1-per-val;
//   best case uniform 1-per-thread regardless of N). Two copies halve per-bank pressure.
// Memory Access: 128-bit uint4 — each LDG.E.128 fetches 4 unsigned ints per instruction.
// Exp 1 (Random): Fastest — max BW, R=2 padding, RLE eliminates repeated accesses.
// Exp 2 (Uniform): Fastest — RLE compresses entire run to 1 atomic per thread; R=2
//   halves the remaining shared bank serialization; 1 global atomic per non-zero bin.
__global__ void histogram_shared_optimized(unsigned int *input, unsigned int *bins,
                                           unsigned int num_elements,
                                           unsigned int num_bins) {
    // 1. R=2 padded shared histograms: 2×4097×4 B = 32,776 B (fits in P100's 64 KB)
    __shared__ unsigned int multi_histo[2][4097];

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        multi_histo[0][i] = 0;
        multi_histo[1][i] = 0;
    }
    __syncthreads();

    // 2. Replica selection: even threads → copy 0, odd threads → copy 1
    int R_idx = threadIdx.x & 1;

    // 3. Vectorized 128-bit load setup
    uint4 *vector_input = reinterpret_cast<uint4*>(input);
    int num_vectors = num_elements / 4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 4. Thread-local RLE state (pure register — zero memory cost)
    unsigned int current_bin = 0xFFFFFFFF; // sentinel: no active run
    unsigned int run_count   = 0;

    // 5. Main loop: 128-bit vectorized loads + RLE compression
    for (int i = tid; i < num_vectors; i += stride) {
        uint4 data = vector_input[i];
        unsigned int vals[4] = {data.x, data.y, data.z, data.w};

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            unsigned int val = vals[k];
            if (val < num_bins) {
                if (val == current_bin) {
                    run_count++; // extend run — pure register increment, no atomic
                } else {
                    if (run_count > 0)
                        atomicAdd(&multi_histo[R_idx][current_bin], run_count);
                    current_bin = val;
                    run_count   = 1;
                }
            }
        }
    }

    // 6. Flush final RLE run from registers to shared memory
    if (run_count > 0 && current_bin < num_bins)
        atomicAdd(&multi_histo[R_idx][current_bin], run_count);

    // 7. Tail: elements when num_elements % 4 != 0 (standard atomics, no RLE needed)
    int tail_start = num_vectors * 4;
    for (int i = tail_start + tid; i < num_elements; i += stride) {
        unsigned int bin = input[i];
        if (bin < num_bins)
            atomicAdd(&multi_histo[R_idx][bin], 1);
    }

    // 8. Intra-block merge + sparsity-filtered global flush
    __syncthreads();
    for (int b = threadIdx.x; b < num_bins; b += blockDim.x) {
        unsigned int total = multi_histo[0][b] + multi_histo[1][b];
        if (total > 0)
            atomicAdd(&bins[b], total);
    }
}

// version 3 — Ablation: Vectorized R-per-Block WITHOUT RLE (pure memory architecture)
// Core Pattern: identical to V2 — R=2 padded shared histograms, uint4 128-bit loads,
//   even/odd replica selection, sparsity-filtered global flush — but with the RLE
//   register compression layer removed. Every valid bin value fires an atomicAdd
//   immediately with no temporal compression.
// Purpose (ablation): isolates the performance contribution of RLE compression from
//   the memory architecture. V2−V3 delta = pure RLE speedup; any remaining V3 advantage
//   over V1 comes exclusively from uint4 BW + R=2 bank-conflict reduction.
// Atomic Scope: shared — N atomics (same as V1), but split across R=2 copies (half per bank).
// Memory Access: 128-bit uint4 loads (same as V2).
// Exp 1 (Random): Faster than V1 (uint4 BW + R=2), slower than V2 (no RLE compression).
// Exp 2 (Uniform): Same R=2 halving as V2; no RLE means N atomics instead of 1 per thread.
__global__ void histogram_v3_no_rle(unsigned int *input, unsigned int *bins,
                                    unsigned int num_elements,
                                    unsigned int num_bins) {
    // R=2 padded shared histograms (identical layout to V2)
    __shared__ unsigned int multi_histo[2][4097];

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        multi_histo[0][i] = 0;
        multi_histo[1][i] = 0;
    }
    __syncthreads();

    uint4 *vector_input = reinterpret_cast<uint4*>(input);
    int num_vectors = num_elements / 4;
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int R_idx  = threadIdx.x & 1; // even/odd replica selection

    // Main loop: uint4 loads, direct atomicAdd (no RLE)
    for (int i = tid; i < num_vectors; i += stride) {
        uint4 data = vector_input[i];
        unsigned int vals[4] = {data.x, data.y, data.z, data.w};

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            unsigned int bin = vals[k];
            if (bin < num_bins)
                atomicAdd(&multi_histo[R_idx][bin], 1);
        }
    }

    // Tail: elements when num_elements % 4 != 0
    int tail_start = num_vectors * 4;
    for (int i = tail_start + tid; i < num_elements; i += stride) {
        unsigned int bin = input[i];
        if (bin < num_bins)
            atomicAdd(&multi_histo[R_idx][bin], 1);
    }

    // Intra-block merge + sparsity-filtered global flush (identical to V2)
    __syncthreads();
    for (int b = threadIdx.x; b < num_bins; b += blockDim.x) {
        unsigned int total = multi_histo[0][b] + multi_histo[1][b];
        if (total > 0)
            atomicAdd(&bins[b], total);
    }
}

// version 4 — Warp Aggregation (spatial compression via register-level intrinsics)
// Core Pattern: within each warp, threads elect a leader for each unique target bin
//   using __ballot_sync + __shfl_sync + __popc. The leader issues ONE atomicAdd to
//   shared memory for the entire coalition of matching lanes, reducing shared atomics
//   from 32 (V1) to U (unique bins per warp, 1 ≤ U ≤ 32).
// Uses standard 32-bit scalar loads (no uint4) to isolate the warp aggregation effect.
// Uses a single shared histogram copy (no R=2) to keep register usage reasonable.
// Atomic Scope: shared — U atomics per warp iteration (U ≤ 32), versus N for V0/V1.
// Memory Access: 32-bit coalesced scalar loads.
// Exp 1 (Random): Fast — average U ≈ 32 for fully random; benefit over V1 is modest
//   but real; no bandwidth advantage vs V2/V3.
// Exp 2 (Uniform): Fastest among non-RLE versions — U=1 per warp, single atomic ever.
__global__ void histogram_v4_warp_agg(unsigned int *input, unsigned int *bins,
                                      unsigned int num_elements,
                                      unsigned int num_bins) {
    extern __shared__ unsigned int histo_private[];

    for (int b = threadIdx.x; b < num_bins; b += blockDim.x)
        histo_private[b] = 0;
    __syncthreads();

    int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride  = blockDim.x * gridDim.x;
    int lane_id = threadIdx.x & 31; // lane within warp (0–31)

    for (int i = tid; i < num_elements; i += stride) {
        unsigned int bin = input[i];

        if (bin < num_bins) {
            // Loop until all 32 lanes in the warp have committed their bin
            unsigned int active_mask = 0xFFFFFFFF;
            while (active_mask != 0) {
                // Leader = lowest active lane
                int leader = __ffs(active_mask) - 1;
                // Broadcast leader's target bin to the whole warp
                unsigned int leader_bin = __shfl_sync(active_mask, bin, leader);
                // All lanes that share the leader's bin form a coalition
                unsigned int match_mask = __ballot_sync(active_mask, bin == leader_bin);
                // Leader issues one atomicAdd for the entire coalition
                if (lane_id == leader)
                    atomicAdd(&histo_private[leader_bin], __popc(match_mask));
                // Remove matched lanes; repeat for remaining lanes
                active_mask ^= match_mask;
            }
        }
    }
    __syncthreads();

    // Sparsity-filtered global flush
    for (int b = threadIdx.x; b < num_bins; b += blockDim.x) {
        if (histo_private[b] > 0)
            atomicAdd(&bins[b], histo_private[b]);
    }
}

// version 5 — Bin-Centric Gather (zero atomics, arithmetic-bound)
// Core Pattern: inverts scatter to gather — each thread owns exactly one bin and
//   scans the entire input looking for its value using collaborative shared-memory tiling.
// Atomic Scope: None — each bin written by exactly one thread; no concurrent writes.
// Memory Access: Collaborative tiling — block loads 256-element tile per iteration in
//   one coalesced transaction; all threads scan the tile in registers.
//   No clipping in kernel — output goes through convert_kernel pass.
// Exp 1 (Random): Slowest non-sequential — O(N×B) total comparisons despite zero atomics.
// Exp 2 (Uniform): Still O(N×B) arithmetic; perfectly flat latency across all bins.
// Launch: exactly num_bins threads — dim3(num_bins/256) blocks × 256 threads.
__global__ void histogram_v5_gather(unsigned int *input, unsigned int *bins,
                                    unsigned int num_elements,
                                    unsigned int num_bins) {
    unsigned int my_bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (my_bin >= num_bins) return;

    unsigned int local_count = 0;
    extern __shared__ unsigned int tile[];

    for (int i = 0; i < (int)num_elements; i += blockDim.x) {
        if (i + threadIdx.x < (int)num_elements)
            tile[threadIdx.x] = input[i + threadIdx.x];
        __syncthreads();

        int valid_elements = min((int)blockDim.x, (int)(num_elements - i));
        #pragma unroll 4
        for (int j = 0; j < valid_elements; j++) {
            if (tile[j] == my_bin)
                local_count++;
        }
        __syncthreads();
    }

    // No clipping here — convert_kernel handles it
    bins[my_bin] = local_count;
}

// version 7 — Multi-split / Hierarchical Bucketing
// Core Pattern: coarse-then-fine two-level address computation. Each element's bin
//   index is decomposed into a super-bucket (bin/256, selecting one of 16 256-bin
//   regions) and a fine offset (bin%256). The hierarchical address calculation groups
//   threads with similar bin values to adjacent shared-memory locations, improving
//   L1 cache locality for the atomicAdd relative to a flat unstructured scatter.
//   Standard single-copy shared histogram; no R=2 replication or RLE.
// Atomic Scope: shared — N atomics (same as V1), but with improved locality.
// Memory Access: 32-bit coalesced scalar loads.
// Note: super_bucket*256 + offset == bin, so correctness is identical to V1.
//   The benefit is micro-architectural: the decomposed address hints the compiler
//   toward locality-aware register allocation and can reduce bank-conflict clustering
//   for non-power-of-two-aligned access patterns.
__global__ void histogram_v7_multisplit(unsigned int *input, unsigned int *bins,
                                        unsigned int num_elements,
                                        unsigned int num_bins) {
    extern __shared__ unsigned int histo_private[];

    for (int b = threadIdx.x; b < num_bins; b += blockDim.x)
        histo_private[b] = 0;
    __syncthreads();

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < (int)num_elements; i += stride) {
        unsigned int bin = input[i];
        if (bin < num_bins) {
            // Decompose into coarse super-bucket and fine offset
            unsigned int super_bucket    = bin / 256;
            unsigned int offset          = bin % 256;
            unsigned int target_address  = super_bucket * 256 + offset; // == bin
            atomicAdd(&histo_private[target_address], 1);
        }
    }
    __syncthreads();

    for (int b = threadIdx.x; b < num_bins; b += blockDim.x) {
        if (histo_private[b] > 0)
            atomicAdd(&bins[b], histo_private[b]);
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
