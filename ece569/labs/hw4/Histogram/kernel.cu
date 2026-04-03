// version 0
// global memory only interleaved version
// include comments describing your approach


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



// version 1
// shared memory privatized version
// include comments describing your approach
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
// version 2
// your method of optimization using shared memory 
// include DETAILED comments describing your approach
// for competition you need to include description of the idea
// where you borrowed the idea from, and how you implmented 
__global__ void histogram_shared_optimized(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
    // naive v2: same shared memory privatization as v1
    // serves as a baseline before further optimization
    extern __shared__ unsigned int histogram_private[];
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
    __syncthreads();
    for (int b = threadIdx.x; b < num_bins; b += blockDim.x) {
        atomicAdd(&bins[b], histogram_private[b]);
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
