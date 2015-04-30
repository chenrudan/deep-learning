/*
 * filename: layer_kernel.cuh
 */

#ifndef LAYER_KERNEL_CUH_
#define LAYER_KERNEL_CUH_

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
			i < (n); \
			i += blockDim.x * gridDim.x)


__global__ void im2col_img(const float* conv_result, float* targets, \
		const int numKernels, const int widthNoChannel, const int width, \
		const int img_size, const int filter_channel, \
		const int img_channel, const int filter_size, const int conv_forward_size, \
		const int conv_stride);

__global__ void im2col_filt(const float* imgs, float* targets, \
		const int numKernels, const int widthNoChannel, const int width, \
		const int heightNoBatch, const int img_size, const int img_channel, \
		const int filter_size, const int conv_forward_size, \
		const int conv_step_size);

__global__ void im2col_conv(const float* imgs, float* targets, \
		const int numKernels, const int widthNoBatch, const int width, \
		const int heightNoChannel, const int img_size, \
		const int img_channel, const int filter_size, const int conv_forward_size, \
		const int conv_step_size);

__global__ void reshape_w(float* un_w, const float* w, \
		const int numKernels, const int filter_size, \
		const int filter_channel, const int img_channel);

__global__ void reshape_In(float* in, const float* un_in, \
		const int numKernels, const int in_size, \
		const int img_channel);

__global__ void reshape_y(const float* un_y_h, float* y_h, const int numKernels, \
		const int conv_forward_size, const int filter_channel);

__global__ void reshape_dE_dx_sigmoid(float* un_dE_dx_h, const float* dE_dx_h, \
		const int numKernels, const int conv_forward_size, \
		const int filter_channel);

__global__ void convolution_forward(const float* imgs, const float* filters, \
		const float* biases, float* targets, const int filConvtimes, \
		const int imgConvtimes);

__global__ void avg_pooling(float* convOutputs, float* targets);

__global__ void compute_dE_dy_j(const float* y_j, const float* labels, \
		float* dE_dy_j, const int width);

__global__ void compute_dE_dy_h_avg(const float* dE_dy_i, float* out);

__global__ void compute_dE_dy_max(float* dE_dy_i, float* out, int* maxPoolPos, \
		const int conv_forward_size, const int pool_forward_size, \
		const int max_pool_size, const int stride);

__global__ void max_pooling(float* convOutputs, float* targets, int* maxPoolPos, \
		const int conv_forward_size, const int pool_forward_size, \
		const int max_pool_size, const int stride);

__global__ void convolution_backward(const float* imgs, const float* filters, \
		float* targets, int filConvtimes, int imgConvtimes);

__global__ void compute_dE_db(const float* dE_dx_h, float* dE_db_h, \
		const int conv_forward_size);

__global__ void compute_dE_dy(const float* y_j, const float* labels, \
		float* dE_dy_j, const int width);













#endif
