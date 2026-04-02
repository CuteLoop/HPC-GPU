// version 0
// global memory only interleaved version
// include comments describing your approach
__global__ void histogram_global_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

// insert your code here

}


// version 1
// shared memory privatized version
// include comments describing your approach
__global__ void histogram_shared_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

// insert your code here

}


// version 2
// your method of optimization using shared memory 
// include DETAILED comments describing your approach
// for competition you need to include description of the idea
// where you borrowed the idea from, and how you implmented 
__global__ void histogram_shared_optimized(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

// insert your code here

}

// clipping function
// resets bins that have value larger than 127 to 127. 
// that is if bin[i]>127 then bin[i]=127

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {


// insert your code here

}
