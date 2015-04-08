/*
 * filename: logistic_kernel.cu
 */

#include <cuda_runtime.h>
#include "logistic_kernel.cuh"

//row-major
__global__ void compute_dE_dy(const float* y_j, const float* labels, \
		float* dE_dy_j, const int width) {
	const int tx = blockIdx.x;
	const int ty = blockIdx.x * width + threadIdx.x;
	
	const int lab = labels[tx];
						   
	if(threadIdx.x < width)
		dE_dy_j[ty] = y_j[ty] - (lab == threadIdx.x);
	__syncthreads();
}



