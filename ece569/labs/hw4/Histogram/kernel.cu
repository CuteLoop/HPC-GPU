// version 0
// global memory only interleaved version
// include comments describing your approach


__global__ void histogram_global_kernel(unsigned char *d_in, unsigned int *d_out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (i < size) {
        int alphabet_postion = buffer[i]-'a';
        if (alphabet_postion >= 0 && alphabet_postion < 26) {
            atomicAdd(&d_out[alphabet_postion/4], 1);
        }
        i += stride;
    }
}



// version 1
// shared memory privatized version
// include comments describing your approach
__global__ void histogram_kernel(unsigned char *bufffer,
     long size, unsigned int *histogram) {
        __shared__ unsigned int histogram_private[7];
        if (threadIdx.x < 7) histogram_private[threadIdx.x] = 0;
        __syncthreads();
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int stride = blockDim.x * gridDim.x;
        while (i< size) {
            position = bufffer[i] -'a';
            atomicAdd(&histogram_private[position/4], 1);
            i += stride;
            // wait for all threads to finish updating the private histogram
            __syncthreads();
            if (threadIdx.x < 7) {
                atomicAdd(&histogram[threadIdx.x], histogram_private[threadIdx.x]);
            }
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

// insert your code here

}

// clipping function
// resets bins that have value larger than 127 to 127. 
// that is if bin[i]>127 then bin[i]=127

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

    // clipping

// insert your code here

}
