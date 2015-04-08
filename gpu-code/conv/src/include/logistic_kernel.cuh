/*
 * filename: logistic_kernel.cuh
 */

#ifndef LOGISTIC_KERNEL_CUH_
#define LOGISTIC_KERNEL_CUH_

__global__ void compute_dE_dy(const float* y_j, const float* labels, \
		float* dE_dy_j, const int width);


#endif
