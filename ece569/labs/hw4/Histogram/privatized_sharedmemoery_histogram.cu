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