#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B using shared memory tiling.

__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use tiling with shared memory for arbitrary size

  // Shared memory tiles for A and B
  __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

  // Row and column of the C element this thread is responsible for
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

  float sum = 0.0f;

  // Number of tiles needed to cover the width of A 
  int numTiles = (numAColumns + TILE_WIDTH - 1) / TILE_WIDTH;

  for (int t = 0; t < numTiles; t++) {
    // Column index into A for this thread's tile element
    int aCol = t * TILE_WIDTH + threadIdx.x;
    // Row index into B for this thread's tile element
    int bRow = t * TILE_WIDTH + threadIdx.y;

    // Load element of A into shared memory
    // Boundary conditions and if out of bounds pad with zero.
    if (row < numARows && aCol < numAColumns) {
      tileA[threadIdx.y][threadIdx.x] = A[row * numAColumns + aCol];
    } else {
      tileA[threadIdx.y][threadIdx.x] = 0.0f;
    }

    // Load element of B into shared memory
    if (bRow < numBRows && col < numBColumns) {
      tileB[threadIdx.y][threadIdx.x] = B[bRow * numBColumns + col];
    } else {
      tileB[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; k++) {
      sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }
    __syncthreads();
  }

  // Write the result to global memory
  if (row < numCRows && col < numCColumns) {
    C[row * numCColumns + col] = sum;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA; // A matrix on device
  float *deviceB; // B matrix on device
  float *deviceC; // C matrix on device
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C
  int numCColumns; // number of columns in the matrix C

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);

  //@@ Set numCRows and numCColumns
  numCRows    = numARows;     
  numCColumns = numBColumns;  
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  wbCheck(cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  wbCheck(cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  // TILE_WIDTH is set to 16 on line 13; block size matches tile size
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid((numCColumns + TILE_WIDTH - 1) / TILE_WIDTH,
               (numCRows + TILE_WIDTH - 1) / TILE_WIDTH);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC,
                                              numARows, numAColumns,
                                              numBRows, numBColumns,
                                              numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  wbCheck(cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  wbCheck(cudaFree(deviceA));
  wbCheck(cudaFree(deviceB));
  wbCheck(cudaFree(deviceC));
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
