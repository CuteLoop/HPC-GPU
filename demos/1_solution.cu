/*
load the cuda module:
$ module load cuda11/11.0

to compile: 
$ nvcc -o myadd add_template.cu

to execute: 
$ ./myadd

this template will compile and run the host side as it is. 
*/

#include<cuda.h>
#include <cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>

// first define the kernel
// later we will add print statement to print thread id and 
// blockid for the 16 blocks and 1 thread/block configuration
// insert your code here 

__global__ void vecAdd(int* k_a, int* k_b, int k_size){

int tid;
tid = blockIdx.x*blockDim.x+threadIdx.x;

if (tid < k_size) {
 k_a[tid] = k_a[tid]+k_b[tid];
//  printf("I am thread %d in block number %d and my global positoin is %d\n", threadIdx.x, blockIdx.x, tid);
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

int count = 10000000;


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



// allocate memory on device, check for failure
// insert your code here 

if (cudaMalloc( (void**)&d_a, count*sizeof(int)) != cudaSuccess) {
  printf("a array allocation error\n");
  return(0);
}

if (cudaMalloc( (void**)&d_b, count*sizeof(int)) != cudaSuccess) {
  printf("b array allocation error\n");
  cudaFree(d_a);
  return(0);
}


// copy data to device, check for failure, free device if needed
// insert your code here 

if (cudaMemcpy(d_a, h_a, count*sizeof(int), cudaMemcpyHostToDevice)!=cudaSuccess){
  printf("host to device a array transfer failure\n");
  cudaFree(d_a);
  cudaFree(d_b);
  return(0);
 }

if (cudaMemcpy(d_b, h_b, count*sizeof(int), cudaMemcpyHostToDevice)!=cudaSuccess){
  printf("host to device b array transfer failure\n");
  cudaFree(d_a);
  cudaFree(d_b);
  return(0);
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
// insert your code here 
dim3 mygrid(ceil(count/256.0),1,1);
dim3 myblock(256);
//dim3 mygrid(16);
//dim3 myblock(2);

//this code measures kernel time. move the start and stop event calls  
//to include time spent on memory allocation or data transfers 
cudaEventRecord(startEvent, 0);

vecAdd<<<mygrid,myblock>>>(d_a,d_b,count);

cudaEventRecord(stopEvent, 0);
cudaEventSynchronize(stopEvent);



//if printing from the kernel flush the printfs 
// insert your code here 
//cudaDeviceSynchronize();

// retrieve data from the device, check for error, free device if needed 
// insert your code here 
if (cudaMemcpy(h_a, d_a, count*sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
{
  printf(" data transfer from devie to host failire\n");
  cudaFree(d_a);
  cudaFree(d_b);
  return(0);
}

 

cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
printf("Total execution time (ms) %f\n",elapsedTime);
for(i=0;i<5;i++)
   printf("%d \n",h_a[i]);
   
cudaEventDestroy(startEvent);
cudaEventDestroy(stopEvent);


return 0;
}

