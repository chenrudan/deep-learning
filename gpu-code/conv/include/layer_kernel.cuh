/*
 * filename: layer_kernel.cuh
 */
#ifndef LAYER_KERNEL_CUH_
#define LAYER_KERNEL_CUH_

#include "param.h"

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
			i < (n); \
			i += blockDim.x * gridDim.x)

__global__ void forward_convolution(const float* x, const float* w, \
		const float* bias, \
		float* targets, const int in_size, const int in_channel, \
		const int out_size, const int filter_size, const int filter_channel, \
		const int stride, const int box_out_size, const int box_num_size);

__global__ void backward_convolution(const float* dE_dy, const float *w, \
		float* targets, \
		const int box_in_size, const int box_out_size, \
		const int out_channel, const int in_channel, \
		const int out_size, const int filter_size, \
		const int stride, const int box_num_size);

__global__ void compute_convolution_derivs(const float* dE_dy, const float *x, \
		float* dE_dw, const int box_in_size, const int box_out_size, \
		const int out_channel, const int in_channel, \
		const int out_size, const int filter_size, \
		const int stride, const int box_num_size);

__global__ void compact_dervis_w(const float* unranged_dE_dw, \
		float* dE_dw, const int filter_size, const int box_num_size, \
		const int minibatch_size, const int in_channel, const int out_channel);

__global__ void compute_derivs_of_bias(const float* dE_dy, float* targets, \
		const int out_size, const int out_channel, \
		const int box_out_size, const int box_num_size);

__global__ void pad_to_ori(float* dst, const float* src, const int num_kernel, \
		const int img_size, const int padded_img_size, const int img_channel);

__global__ void ori_to_padding(const float* src, float* dst, const int numKernels, \
        const int img_size, const int padded_img_size, const int img_channel);

__global__ void reshape_dE_dy2(float* un_dE_dy, const float* dE_dy, \
		const int numKernels, const int conv_forward_size, \
		const int filter_channel);

__global__ void reshape_dE_db_tmp(float* dst, const float* ori, \
		const int numKernels, const int filter_channel);

__global__ void compute_dE_dy_max(const float* dE_dy_i, float* targets, \
		int* maxPoolPos, \
		const int box_in_size, const int box_out_size, \
		const int in_channels, \
		const int pool_forward_size, const int max_pool_size, \
		const int stride, const int box_num_size);

__global__ void compute_dE_dy_avg(const float* dE_dy_i, float* targets, \
		const int box_in_size, const int box_out_size, \
		const int in_channels, \
		const int pool_forward_size, const int avg_pool_size, \
		const int stride, const int box_num_size);

__global__ void avg_pooling(const float* convOutputs, float* targets, \
		const int box_in_size, const int in_channels, \
		const int pool_forward_size, const int avg_pool_size, \
		const int stride, const int box_out_size, const int box_num_size);

__global__ void max_pooling(const float* convOutputs, float* targets, int* maxPoolPos, \
		const int conv_forward_size, const int in_channels, \
		const int pool_forward_size, const int max_pool_size, \
		const int stride, \
		const int box_out_size, const int box_num_size);

__global__ void compute_dE_db(const float* dE_dy, float* dE_db_h, \
		const int conv_forward_size);

__global__ void compute_dE_dy(const float* y_j, const int* labels, \
		float* dE_dy_j, const int width);

__global__ void compactOverlap(float* src, float* targets, \
		const int in_size, const int in_channel, const int overlap_len, \
		const int box_in_size, const int box_num_size);












#endif
