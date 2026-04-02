/*Upgrade from basic histogram
strided access for coalesced access

*/ 



__global__ void histo_kernel(unsigned char *d_in, unsigned int *d_out, int size) {
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
