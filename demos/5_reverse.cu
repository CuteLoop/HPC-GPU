/*
source code for experimenting with reverse function covered in Module 19 
you will write the baseline, coalesced and dynamic versions


load the cuda module for ocelote:
$ module load cuda11/11.0


load the cuda module for ElGato:
$module load cuda11/11.0


to compile: 
$ nvcc -o reverse reverse.cu


to execute: 
$ ./reverse

this template will compile and run the host side as it is. 

exit out of interactive session before running the next 
interactive job or compiling: 

*/

#include <stdio.h>
#include<cuda.h>
#include <cuda_runtime.h>
#include<stdlib.h>


__global__ void baselineReverse(int *d, int n)
{
  
  __shared__ int s[64];
  
  int id = threadIdx.x;
  s[id] = d[id];
  __syncthreads();
  d[n-id-1] = s[id];
    __syncthreads();
  
}

__global__ void coalescedReverse(int *d, int n)
{
  // write coalesced memory access version here
 
}

__global__ void dynamicReverse(int *d, int n)
{
  
}

int main(void)
{
  const int n = 64;
  int a[n], r[n], d[n];
  
  cudaEvent_t startEvent, stopEvent;
  float elapsedTime;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);
  srand(time(NULL));
  
  for (int i = 0; i < n; i++) {
    a[i] = i;
    r[i] = n-i-1;
    d[i] = 0;
  }

  int *d_d;
  if (cudaMalloc(&d_d, n * sizeof(int))!= cudaSuccess) {
     printf("malloc error for d_a\n");
 return 0;
 } 
  
  // run version with static baseline shared memory
if (cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice)!= cudaSuccess) {
 printf("data transfer error from host to device on d_d\n");
  return 0;
 }
  
  cudaEventRecord(startEvent, 0);
  
  baselineReverse<<<1,n>>>(d_d, n);

 cudaError_t err = cudaGetLastError();        // Get error code

   if ( err != cudaSuccess )
   {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
      exit(-1);
   }



  cudaDeviceSynchronize();
  
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
  printf("Baseline static kernel execution time (ms) %f\n",elapsedTime);
  
  if (cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
  printf("data transfer error from device to host on d\n");
  return 0;
 }
    
  for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
 
 
  // run version with static coalesced shared memory
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  
  cudaEventRecord(startEvent, 0);
  
  coalescedReverse<<<1,n>>>(d_d, n);
  
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
  printf("Coalesced static kernel execution time (ms) %f\n",elapsedTime);
  
  cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
 // for (int i = 0; i < n; i++) 
 //   if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
	
  

  // run version with dynamic coalesced shared memory
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  
  cudaEventRecord(startEvent, 0);
  
  dynamicReverse<<<1,n,n*sizeof(int)>>>(d_d, n);
  
  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
  printf("Coalesced dynamic kernel execution time (ms) %f\n",elapsedTime);
  
  cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
  // for (int i = 0; i < n; i++) 
  //  if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
  
  
   
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}
