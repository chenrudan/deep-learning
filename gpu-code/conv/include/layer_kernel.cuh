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

__global__ void ori_to_padding(const float* src, float* dst, const int numKernels, \
        const int img_size, const int padded_img_size, const int img_channel);

__global__ void im2col_img(const float* conv_result, float* targets, \
		const int numKernels, const int img_size, const int filter_channel, \
		const int img_channel, const int filter_size, const int conv_forward_size, \
		const int conv_stride);

__global__ void im2col_filt(const float* imgs, float* targets, \
		const int numKernels, const int img_size, const int img_channel, \
		const int filter_size, const int conv_forward_size, \
		const int conv_step_size);

__global__ void im2col_conv(const float* imgs, float* targets, \
		const int numKernels, const int minibatch, const int img_size, \
		const int img_channel, const int filter_size, const int conv_forward_size, \
		const int conv_step_size);

__global__ void reshape_w(float* un_w, const float* w, \
		const int numKernels, const int filter_size, \
		const int filter_channel, const int img_channel);

__global__ void reshape_In(float* in, const float* un_in, \
		const int numKernels, const int in_size, \
		const int padded_img_size, const int img_channel);

__global__ void reshape_y(const float* un_y, float* y, const int numKernels, \
		const int conv_forward_size, const int filter_channel);

__global__ void reshape_dE_dy(float* un_dE_dy, const float* dE_dy, \
		const int numKernels, const int conv_forward_size, \
		const int filter_channel);

__global__ void reshape_dE_dy2(float* un_dE_dy, const float* dE_dy, \
		const int numKernels, const int conv_forward_size, \
		const int filter_channel);

__global__ void reshape_dE_db_tmp(float* dst, const float* ori, \
		const int numKernels, const int filter_channel);

__global__ void convolution_forward(const float* imgs, const float* filters, \
		const float* biases, float* targets, const int filConvtimes, \
		const int imgConvtimes);

__global__ void compute_dE_dy_j(const float* y_j, const float* labels, \
		float* dE_dy_j, const int width);

__global__ void compute_dE_dy_avg(const float* dE_dy_i, float* out);

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

__global__ void convolution_backward(const float* imgs, const float* filters, \
		float* targets, int filConvtimes, int imgConvtimes);

__global__ void compute_dE_db(const float* dE_dy, float* dE_db_h, \
		const int conv_forward_size);

__global__ void compute_dE_dy(const float* y_j, const float* labels, \
		float* dE_dy_j, const int width);

__global__ void compactOverlap(float* src, float* targets, \
		const int in_size, const int com_stride, const int com_len, \
		const int out_size, const int out_channel);












#endif
