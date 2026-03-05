

__global__ void saxpy(int n, float a, float *x, float *y) {
  //@@ Insert code to implement saxpy here
  //@@ You should use a 1D grid of 1D blocks
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < n) {
    y[id] = a * x[id] + y[id];
  }
}


int main(int argc, char** argv){

    
    size_t inputLength;
    float *hostInput1;
    float *hostInput2;
   
    float *deviceInput1;
    float *deviceInput2;

    // Allocate host memory
    hostInput1 = (float *)malloc(inputLength * sizeof(float));
    hostInput2 = (float *)malloc(inputLength * sizeof(float));
    // Allocate device memory
    cudaMalloc((void **)&deviceInput1, inputLength * sizeof(float));
    cudaMalloc((void **)&deviceInput2, inputLength * sizeof(float));
    // Copy data from host to device
    cudaMemcpy(deviceInput1,hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);

    // run kernel
    // pre run: do grid block calculations// good number divides 1024 and 2048
    // good number divides 1024 and 2048
    int blockSize = 256;
    // blocks needed cieling of inputLength / blockSize 
    int numBlocks = (inputLength + blockSize - 1) / blockSize; 
    // run
    saxpy<<<numBlocks, blockSize>>>(inputLength, 2.0f, deviceInput1, deviceInput2);
    cudaDeviceSynchronize();
    // Copy data from device to host
    cudaMemcpy(hostInput2, deviceInput2, inputLength * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    free(hostInput1);
    free(hostInput2);

    return 0;
}


    


