/*
Template source code for experimenting with reduction functions covered in Modules 33-34 

load the cuda module:
$ module load cuda11/11.0

to compile: 
$ nvcc -o reduce reduction_template.cu

to execute: 
$ ./reduce > out.txt
*/


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

__global__ void global_reduce_stride(float * d_out, float * d_in)
{
    
    
}

__global__ void shared_reduce_stride(float * d_out, float * d_in)
{
   
}

__global__ void shared_reduce_stride_nodiverge(float * d_out, float * d_in)
{
    
}


__global__ void shared_reduce_reverse(float * d_out, const float * d_in)
{
   
}

__global__ void shared_reverse_firstreduction(float * d_out, const float * d_in)
{
    
}


void reduce(float * d_out, float * d_intermediate, float * d_in, 
            int size, int version)
{
    // assumes that size is not greater than maxThreadsPerBlock^2
    // and that size is a multiple of maxThreadsPerBlock
    const int maxThreadsPerBlock = 1024;
    int threads = maxThreadsPerBlock;
    int blocks = size / maxThreadsPerBlock;
    
    if (version==4)
    {
        global_reduce_stride<<<blocks, threads>>>(d_intermediate, d_in);
    }
    else if (version==3)
    {
        shared_reverse_firstreduction<<<blocks/2, threads, threads * sizeof(float)>>>(d_intermediate, d_in);
    }
    else if (version==2)
    {
        shared_reduce_reverse<<<blocks, threads, threads * sizeof(float)>>>(d_intermediate, d_in);
    }
    else if (version==1)
    {
        shared_reduce_stride_nodiverge<<<blocks, threads, threads * sizeof(float)>>>(d_intermediate, d_in);
    }
    else if (version==0)
    {
        shared_reduce_stride<<<blocks, threads, threads * sizeof(float)>>>(d_intermediate, d_in);
    }
    // now we're down to one block left, so reduce it
    threads = blocks; // launch one thread for each block in prev step
    blocks = 1;
     
    if (version==4)
    {
        global_reduce_stride<<<blocks, threads>>>(d_out, d_intermediate);
    }
    else if (version==3)
    {
        shared_reverse_firstreduction<<<blocks/2, threads, threads * sizeof(float)>>>(d_out, d_intermediate);
    }
    else if (version==2)
    {
        shared_reduce_reverse<<<blocks, threads, threads * sizeof(float)>>>(d_out, d_intermediate);
    }
    else if (version == 1)
    {
       shared_reduce_stride_nodiverge<<<blocks, threads, threads * sizeof(float)>>>(d_out, d_intermediate);
        
    }
    else if (version == 0)
    {
        shared_reduce_stride<<<blocks, threads, threads * sizeof(float)>>>(d_out, d_intermediate);
    }
}

int main(int argc, char **argv)
{

    struct timeval st, et;
    int j;
    float nsum;
    int elapsed;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp devProps;

    const int ARRAY_SIZE = 1 << 20;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // generate the input array on the host
    float h_in[ARRAY_SIZE];
    float sum = 0.0f;
    for(int i = 0; i < ARRAY_SIZE; i++) {
        // generate random float in [-1.0f, 1.0f]
        h_in[i] = -1.0f + (float)random()/((float)RAND_MAX/2.0f);
        sum += h_in[i];
    }


    // declare GPU memory pointers
    float * d_in, * d_intermediate, * d_out;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_intermediate, ARRAY_BYTES); // overallocated
    cudaMalloc((void **) &d_out, sizeof(float));

    // transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 

    int whichKernel = 0;      
    cudaEvent_t start, stop;
  
    nsum=0.0;
    gettimeofday(&st,NULL);
    for (j=0; j<ARRAY_SIZE;j++) {
        nsum = nsum + h_in[j];
     }
     gettimeofday(&et, NULL);

     elapsed = ((et.tv_sec - st.tv_sec) * 1000000) + (et.tv_usec - st.tv_usec);
     printf("serial code execution time %f (ms)\n" ,elapsed/1000.0);
     printf("--------------------------\n");
     printf("\n");

     if (cudaGetDeviceProperties(&devProps, dev) == 0)
     {
         printf("Using device %d:\n", dev);
         printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
         devProps.name, (int)devProps.totalGlobalMem, 
            (int)devProps.major, (int)devProps.minor, 
            (int)devProps.clockRate);
     }
  
  for(whichKernel=0; whichKernel<5; whichKernel++) 
  { 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // launch the kernel
   
    switch(whichKernel) {
    case 0:
        printf("Running shared stride reduce\n");
        cudaEventRecord(start, 0);
        for (int i = 0; i < 100; i++)
        {
            reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, 0);
        }
        cudaEventRecord(stop, 0);
        break;
    case 1:
        printf("Running shared stride no divergent reduce\n");
        cudaEventRecord(start, 0);
        for (int i = 0; i < 100; i++)
        {
            reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, 1);
        }
        cudaEventRecord(stop, 0);
        break;
    case 2:
        printf("Running shared reduce reversed\n");
        cudaEventRecord(start, 0);
        for (int i = 0; i < 100; i++)
        {
            reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, 2);
        }
        cudaEventRecord(stop, 0);
        break;
    case 3:
        printf("Running global reduce stride - naive first reduction\n");
        cudaEventRecord(start, 0);
        for (int i = 0; i < 100; i++)
        {
            reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, 3);
        }
        cudaEventRecord(stop, 0);
        break;
    case 4:
        printf("Running global reduce stride - naive\n");
        cudaEventRecord(start, 0);
        for (int i = 0; i < 100; i++)
        {
            reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, 4);
        }
        cudaEventRecord(stop, 0);
        break;
    default:
        fprintf(stderr, "error: ran no kernel\n");
        exit(EXIT_FAILURE);
    }
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);    
    elapsedTime /= 100.0f;      // 100 trials

    // copy back the sum from GPU
    float h_out;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("average time elapsed >>>>>>> %f\n", elapsedTime);
    printf("\n");
  } 
  // end of for loop of kernel versions
    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_intermediate);
    cudaFree(d_out);
        
    return 0;
}
