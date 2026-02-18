/*
source code for experimenting with average  function covered in Module 17

load the cuda module:
$ module load cuda11/11.0

to compile: 
$ nvcc -o average average.cu


to execute: 
$ ./average > out.txt


*/
// Using different memory spaces in CUDA
#include <stdio.h>
#include<cuda.h>
#include <cuda_runtime.h>
#include<stdlib.h>
/**********************
 * using local memory *
 **********************/



/**********************
 * using global memory *
 **********************/

// a __global__ function runs on the GPU & can be called from host
__global__ void global_memory_average(int *array, float* average, int size)
{
        // local variables, private to each thread
    int i, index = threadIdx.x;
    float avg = 0.0f; 
    int sum = 0;

   
    for (i=0; i<index; i++) { sum += array[i]; }
    
    if (index>0)
      avg = sum/(index+0.0f);

    average[index] = avg;
}

/**********************
 * using shared memory *
 **********************/

// (for clarity, hardcoding 128 threads/elements and omitting out-of-bounds checks)
__global__ void shared_memory_average(int *array, float* average, int size)
{
    // local variables, private to each thread
    int i, index = threadIdx.x;
    float avg =0.0f;
    int sum = 0;

    // __shared__ variables are visible to all threads in the thread block
    // and have the same lifetime as the thread block
    __shared__ float sh_arr[128];

    // copy data from "array" in global memory to sh_arr in shared memory.
    // here, each thread is responsible for copying a single element.
    sh_arr[index] = array[index];

    __syncthreads();    // ensure all the writes to shared memory have completed

    // now, sh_arr is fully populated. Let's find the average of all previous elements
    for (i=0; i<index; i++) { sum += sh_arr[i]; }
    
    if (index>0)
    avg = sum / (index+0.0f);

    // if array[index] is greater than the average of array[0..index-1], replace with average.
    // since array[] is in global memory, this change will be seen by the host (and potentially 
    // other thread blocks, if any)
    average[index] = avg; 

    
}

int main(int argc, char **argv)
{
  const int n = 128;
  int a[n]; 
  float d[n];
  
  cudaEvent_t startEvent, stopEvent;
  float elapsedTime;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);
  srand(time(NULL));
  
  for (int i = 0; i < n; i++) {
    a[i] = i+1;
    d[i] = i+1.0f;
  }

  int *d_d;
  cudaMalloc(&d_d, n * sizeof(int)); 
  
  float *d_average;
  cudaMalloc(&d_average, n * sizeof(float)); 
  // run version with static baseline shared memory
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_average, d, n*sizeof(float), cudaMemcpyHostToDevice);
  
  cudaEventRecord(startEvent, 0);
  global_memory_average<<<1,n>>>(d_d,d_average, n);
  
  
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
  printf("Baseline global only kernel execution time (ms) %f\n",elapsedTime);
  
  cudaMemcpy(d, d_average, n*sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
    printf(" %d ",a[i]);

   printf("\n");
   
  for (int i = 0; i < n; i++) 
    printf(" %f ",d[i]);

   printf("\n");
    
  // run version with shared memory
  //cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  for (int i = 0; i < n; i++) {
    a[i] = i+1;
    d[i] = i+1.0f;
  }
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_average, d, n*sizeof(float), cudaMemcpyHostToDevice);
  cudaEventRecord(startEvent, 0);
  
  shared_memory_average<<<1,n>>>(d_d,d_average, n);
  
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
  printf("Shared memory kernel execution time (ms) %f\n",elapsedTime);
  
  cudaMemcpy(d, d_average, n*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
    printf(" %d ",a[i]);

   printf("\n");
   
  for (int i = 0; i < n; i++) 
    printf(" %f ",d[i]);

   printf("\n");
    
  
   
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    

    
}
