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

// allocate memory on device, check for failure
// insert your code here 



// copy data to device, check for failure, free device if needed
// insert your code here 


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




//if printing from the kernel flush the printfs 
// insert your code here 


// retrieve data from the device, check for error, free device if needed 
// insert your code here 

 
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

