/*
In this exercise you will fix bug in the vector addtion code using cuda-memcheck. 

load the cuda module:
$ module load cuda11/11.0

to compile: 
$ nvcc -o myadd add.cu

to execute: 
$ ./myadd

what do you observe? 


Now run the same executable with cuda-memcheck
$ cuda-memcheck ./myadd

now recompile with the following options:
$ nvcc -o myadd add.cu –Xcompiler –rdynamic –lineinfo

run with cuda-memcheck
$ cuda-memcheck ./myadd

*/


#include<cuda.h>
#include <cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>

// define the kernel 
__global__ void AddInts(int * k_a, int* k_b, int k_count){

int  tid;

tid =  blockDim.x* blockIdx.x + threadIdx.x;


 // print thread id and blockid for the 16 blocks and 1 thread/block configuration
 //printf("my thread id is %d and block id is %d\n",threadIdx.x, blockIdx.x);
 k_a[tid] = k_a[tid]+k_b[tid];
 

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
printf("before addition\n");
for(i=0;i<5;i++)
   printf("%d and %d\n",h_a[i],h_b[i]);

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

/* 
generic kernel launch: 
b: blocks
t: threads
shmem: amount of shard memory allocated per block, 0 if not defined

AddInts<<<dim3(bx,by,bz), dims(tx,ty,tz),shmem>>>(parameters)
dim3(w,1,1) = dim3(w) = w

AddInts<<<dim3(4,4,2),dim3(8,8)>>>(....)

How many blocks?
How many threads/blocks?
How many threads?

*/

/* 
 1) set the grid size and block size with the dim3 structure and launch the kernel 
 intitially set the block size to 256 and determine the grid size 
 launch the kernel
 
 2) later we will experiment with printing block ids for the configuration of
 16 blocks and 1 thread per block. For this second experiment insert printf statement 
 in the kernel. you will need cudaDeviceSynchronize() call after kernel launch to 
 flush the printfs. 
 
*/
dim3 mygrid(ceil(count/256.0));
dim3 myblock(256);

//dim3 mygrid(16);
//dim3 myblock(1);
AddInts<<<mygrid,myblock>>>(d_a,d_b,count);

//if printing from the kernel flush the printfs 
//cudaDeviceSynchronize();


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
for(i=0;i<5;i++)
   printf("%d \n",h_a[i]);
   
cudaEventDestroy(startEvent);
cudaEventDestroy(stopEvent);


return 0;
}

