/*
 * filename: convnet_kernel.cuh
 */

#ifndef CONVNET_KERNEL_CUH_
#define CONVNET_KERNEL_CUH_

#define IMG_CHANNEL				3
#define IMG_SIZE				32
#define FILTER_SIZE				9
#define FILTER_CHANNEL			16
#define CONV_STEP_SIZE			1
#define CONV_FORWARD_SIZE   	24
#define POOL_FORWARD_SIZE   	6
#define AVG_POOL_X				4
#define AVG_POOL_Y				4
#define MAX_POOL_X				2
#define MAX_POOL_Y				2


__global__ void im2col_filt(const float* imgs, float* targets, \
                const int numKernels, const int widthNoChannel, const int width, \
                const int heightNoChannel);

__global__ void im2col_conv(const float* imgs, float* targets, \
                const int numKernels, const int widthNoBatch, const int widthNoChannel, \
                const int width, const int height);

__global__ void reshape_y_h(const float* un_y_h, float* y_h, const int numKernels);

__global__ void reshape_dE_dx_h(float* un_dE_dx_h, const float* dE_dx_h, \
                        const int numKernels);

__global__ void convolution_forward(const float* imgs, const float* filters, \
         const float* biases, float* targets, const int filConvtimes, \
		 const int imgConvtimes);

__global__ void avg_pooling(float* convOutputs, float* targets);

__global__ void compute_dE_dy_j(const float* y_j, const float* labels, \
		float* dE_dy_j, const int width);

__global__ void compute_dE_dy_h_avg(const float* dE_dy_i, float* out);

__global__ void compute_dE_dy_h_max(float* dE_dy_i, float* out, int* maxPoolPos);

__global__ void max_pooling(float* convOutputs, float* targets, int* maxPoolPos);

__global__ void convolution_backward(const float* imgs, const float* filters, \
				float* targets, int filConvtimes, int imgConvtimes);

__global__ void compute_dE_db_h(const float* dE_dx_h, float* dE_db_h);














#endif
