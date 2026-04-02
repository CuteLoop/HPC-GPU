#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include<stdlib.h>

using namespace std;

__global__ void mykernel1(unsigned long long* time)
{
  __shared__ float shared[1024];

  // clock returns clock ticks
  unsigned long long startTime = clock();
 
  //all threads are accessing the same location (broadcast) 
  shared[0]++;

  unsigned long long finishTime = clock();
  *time = (finishTime-startTime);
}

__global__ void mykernel2(unsigned long long* time)
{
  __shared__ float shared[1024];
  unsigned long long startTime = clock();


  // no bank conflict
  shared[threadIdx.x]++;

  unsigned long long finishTime = clock();
  *time = (finishTime-startTime);
}

__global__ void mykernel3(unsigned long long* time)
{
  __shared__ float shared[1024];
  unsigned long long startTime = clock();

  shared[threadIdx.x*4]++;

  unsigned long long finishTime = clock();
  *time = (finishTime-startTime);
}

__global__ void mykernel4(unsigned long long* time)
{
  __shared__ float shared[1024];
  unsigned long long startTime = clock();

  shared[threadIdx.x*8]++;

  unsigned long long finishTime = clock();
  *time = (finishTime-startTime);
}


__global__ void mykernel5(unsigned long long* time)
{
  __shared__ float shared[1024];
  unsigned long long startTime = clock();

  shared[threadIdx.x*32]++;

  unsigned long long finishTime = clock();
  *time = (finishTime-startTime);
}



int main()

{
  unsigned long long time;
  unsigned long long* d_time;
  cudaMalloc(&d_time, sizeof(unsigned long long));
 

    mykernel1<<<1,32>>>(d_time);
    cudaMemcpy(&time, d_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    printf("Time for shared[0]: %d\n",(time-14)/32);

    
    mykernel2<<<1,32>>>(d_time);
    cudaMemcpy(&time, d_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    printf("Time for shared[threadIdx.x]: %d\n",(time-14)/32);


    mykernel3<<<1,32>>>(d_time);
    cudaMemcpy(&time, d_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    printf("Time for shared[threadIdx.x*4]: %d\n",(time-14)/32);


    mykernel4<<<1,32>>>(d_time);
    cudaMemcpy(&time, d_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    printf("Time for shared[threadIdx.x*8] : %d\n",(time-14)/32);


    mykernel5<<<1,32>>>(d_time);
    cudaMemcpy(&time, d_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    printf("Time for  shared[threadIdx.x*32]: %d\n",(time-14)/32);
 

  cudaFree(d_time);

  //needed  when you want to use profiler
  cudaDeviceReset();
  return 0;
}
 
