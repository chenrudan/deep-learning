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
		const float* bias, float* targets, \
		const int in_height, const int in_width, const int in_channel, \
		const int out_height, const int out_width, \
		const int filter_height, const int filter_width, const int filter_channel, \
		const int stride_height, const int stride_width, \
		const int box_num_height, const int box_num_width, \
		const int box_in_height, const int box_in_width, \
		const int box_out_height, const int box_out_width);


__global__ void backward_convolution(const float* dE_dy, const float *w, \
		float* targets, \
		const int box_in_height, const int box_in_width, \
		const int box_out_height, const int box_out_width, \
		const int out_channel, const int in_channel, \
		const int out_height, const int out_width, \
		const int filter_height, const int filter_width, \
		const int stride_height, const int stride_width, \
		const int box_num_height, const int box_num_width);


__global__ void compute_convolution_derivs(const float* dE_dy, const float *x, \
		float* dE_dw, const int box_out_height, const int box_out_width, \
		const int out_channel, const int in_channel, const int in_height, \
		const int in_width, const int out_height, const int out_width, \
		const int filter_height, const int filter_width, \
		const int stride_height, const int stride_width, \
		const int box_num_height, const int box_num_width);


__global__ void compact_dervis_w(const float* unranged_dE_dw, \
		float* dE_dw, const int filter_height, const int filter_width, \
		const int box_num_height, const int box_num_width, \
		const int minibatch_size, const int in_channel, const int out_channel);

__global__ void compute_derivs_of_bias(const float* dE_dy, float* targets, \
		const int out_height, const int out_width, const int out_channel, \
		const int box_out_height, const int box_out_width, \
		const int box_num_height, const int box_num_width);


__global__ void pad_to_ori(float* dst, const float* src, const int num_kernel, \
		const int img_height, const int img_width, \
		const int padded_img_height, const int padded_img_width, \
		const int img_channel);

__global__ void ori_to_padding(const float* src, float* dst, const int num_kernel, \
		const int img_height, const int img_width, const int padded_img_height, \
		const int padded_img_width, const int img_channel);

__global__ void max_pooling(const float* convOutputs, float* targets, int* maxPoolPos, \
		const int in_height, const int in_width, \
		const int in_channels, const int out_height, const int out_width, \
		const int filter_height, const int filter_width, \
		const int stride_height, const int stride_width,  \
		const int box_out_height, const int box_out_width, \
		const int box_num_height, const int box_num_width);

__global__ void avg_pooling(const float* convOutputs, float* targets, \
		const int in_height, const int in_width, \
		const int in_channels, const int out_height, const int out_width, \
		const int filter_height, const int filter_width, \
		const int stride_height, const int stride_width,  \
		const int box_out_height, const int box_out_width, \
		const int box_num_height, const int box_num_width);

__global__ void compute_dE_dy_max(const float* dE_dy_i, float* targets, \
		int* maxPoolPos, \
		const int box_in_height, const int box_in_width, \
		const int box_out_height, const int box_out_width, \
		const int num_filters, \
		const int out_height, const int out_width, \
		const int filter_height, const int filter_width, \
		const int stride_height, const int stride_width,  \
		const int box_num_height, const int box_num_width);

__global__ void compute_dE_dy_avg(const float* dE_dy_i, float* targets, \
		const int box_in_height, const int box_in_width, \
		const int box_out_height, const int box_out_width, \
		const int num_filters, \
		const int out_height, const int out_width, \
		const int filter_height, const int filter_width, \
		const int stride_height, const int stride_width,  \
		const int box_num_height, const int box_num_width);

__global__ void compute_dE_dy(const float* y_j, const int* labels, \
		float* dE_dy_j, const int width);


__global__ void compactOverlap(float* src, float* targets, \
		const int in_height, const int in_width, const int in_channel, \
		const int overlap_height, const int overlap_width, \
		const int box_in_height, const int box_in_width, \
		const int box_num_height, const int box_num_width);











#endif
