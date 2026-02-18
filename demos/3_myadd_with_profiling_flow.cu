/*
load the cuda module:
$ module load cuda11/11.0

to compile: 
$ nvcc -o myadd myadd_with_profiling_flow.cu

to execute: 
$ ./myadd

paths for binaries: 
/opt/ohpc/pub/apps/cuda/cuda11/11.0/bin/nvprof
/opt/ohpc/pub/apps/cuda/cuda11/11.0/bin/cuda-memcheck


commands
nvprof --log-file profile.txt ./myadd
nvprof -o profile.timeline ./myadd
nvprof --export-profile profiledata ./myadd

next two commands face permission issue when used on HPC
$nvprof --metrics all -o profile.nvvp ./myadd
$nvprof --metrics all --log-file profile.txt ./myadd

these are the files to use with "nvvp"

*/


#include<cuda.h>
#include <cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>

// define the kernel 
__global__ void AddInts(int * k_a, int* k_b, int k_count){

 int  tid;

 tid =  blockDim.x* blockIdx.x + threadIdx.x;

 if (tid < k_count) {
   // print thread id, blockIDx, threadIdx 
  //  printf("my tid = %d blockIDx = %d threadIdx = %d\n",tid, blockIdx.x, threadIdx.x);
   k_a[tid] = k_a[tid]+k_b[tid];
 }

}

int main()
{
 int i;
 int* d_a;
 int* d_b;

 int* h_a;
 int* h_b;

 cudaEvent_t startEvent, stopEvent;
 float elapsedTime;
 cudaEventCreate(&startEvent);
 cudaEventCreate(&stopEvent);

 int count = 1000;


 srand(time(NULL));


 h_a = (int*)malloc(count*sizeof(int));
 h_b = (int*)malloc(count*sizeof(int));

 for (i=0;i<count;i++) {
  h_a[i] = rand()%1000;
  h_b[i] = rand()%1000;
 }
 //printf("before addition\n");
 //for(i=0;i<5;i++)
  // printf("%d and %d\n",h_a[i],h_b[i]);

 cudaEventRecord(startEvent, 0);

 /* allocate memory on device, check for failure */
 if (cudaMalloc((void **) &d_a, count*sizeof(int)) != cudaSuccess) {
   printf("malloc error for d_a\n");
   return 0;
 }
 
 if (cudaMalloc((void **) &d_b, count*sizeof(int)) != cudaSuccess) {
   printf("malloc error for d_b\n");
   cudaFree(d_a);
   return 0;
 }


 /* copy data to device, check for failure, free device if needed */
 if (cudaMemcpy(d_a,h_a,count*sizeof(int),cudaMemcpyHostToDevice) != cudaSuccess){
  cudaFree(d_a);
  cudaFree(d_b);
  printf("data transfer error from host to device on d_a\n");
  return 0;
 }
 if (cudaMemcpy(d_b,h_b,count*sizeof(int),cudaMemcpyHostToDevice) != cudaSuccess){
  cudaFree(d_a);
  cudaFree(d_b);
  printf("data transfer error from host to device on d_b\n");
  return 0;
 }


 dim3 mygrid(ceil(count/256.0));
 dim3 myblock(256);

 AddInts<<<mygrid,myblock>>>(d_a,d_b,count);

 //if printing from the kernel flush the printfs 
 cudaDeviceSynchronize();


// retrieve data from the device, check for error, free device if needed 
 if (cudaMemcpy(h_a,d_a,count*sizeof(int),cudaMemcpyDeviceToHost) != cudaSuccess){
  cudaFree(d_a);
  cudaFree(d_b);
  printf("data transfer error from host to device on d_a\n");
  return 0;
 }
 
 cudaEventRecord(stopEvent, 0);
 cudaEventSynchronize(stopEvent);
 cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
 printf("Total execution time (ms) %f\n",elapsedTime);
 //for(i=0;i<5;i++)
 //  printf("%d \n",h_a[i]);
   
 cudaEventDestroy(startEvent);
 cudaEventDestroy(stopEvent);


 return 0;
}
