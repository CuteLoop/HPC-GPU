/* Basic GPU histogram implementation

partition input into sections

Each thread takes a section of the input
each thread iterates trough its section
for each element increcment the corresponding bin counter
use atomic add to build histogram in global memory
*/

__global__ void histo_kernel(unsigned char *buffer,
    long size, unsigned int *histo)
    //thread mapping to an index
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        //section size is workload per thread 
        int section_size = (size + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x);
        //start and end index for each thread
        int start = i * section_size;

        for (int k=0; k<section_size; k++) {
            if (start +k< size) 
                // Find alphabet position
                int alpabet_position = buffer[start + k] - 'a';

            if (alphabet_position >= 0 && alpabet_position < 26) 
                // Increment the corresponding bin counter using atomic add
                atomicAdd(&histo[alpabet_position/4], 1);

            }
            
    }

