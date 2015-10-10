/*
 * filename: layer_kernel.cu
 */

#include <cuda_runtime.h>
#include "layer_kernel.cuh"

//struct conv{
//	float* data;	
//};

__device__ float sigmoid(float x) {
	if(x < -300)
		return 0;
	else if( x > 300)
		return 1;
	else
		return 1 / (1 + __expf(-x));
}

__global__ void forward_convolution(const float* x, const float* w, \
		const float* bias, float* targets, \
		const int in_height, const int in_width, const int in_channel, \
		const int out_height, const int out_width, \
		const int filter_height, const int filter_width, const int filter_channel, \
		const int stride_height, const int stride_width, \
		const int box_num_height, const int box_num_width){

	const int num_box = box_num_height * box_num_width;	
	const int img_idx = blockIdx.x;
	const int filt_idx = blockIdx.y / num_box;
	const int box_idx = blockIdx.y % num_box; 
	const int box_row_idx = box_idx / box_num_width;
	const int box_col_idx = box_idx % box_num_width;

	int in_pixs = in_height * in_width;
	int out_pixs = out_height * out_width;
	int filt_pixs = filter_height * filter_width;

	//输出的行列idx，当输出大于MAX_THREAD_SIZE的时候每个线程都做了计算
	int out_row = MAX_THREAD_SIZE * box_row_idx + threadIdx.y;
	int out_col = MAX_THREAD_SIZE * box_col_idx + threadIdx.x;

	if(out_row < out_height && out_col < out_width){
		//x定位哪一个batch
		x += img_idx * in_channel * in_pixs;
		//定位到哪一个filter
		w += filt_idx * in_channel* filt_pixs;
		bias += filt_idx;
		//输出定位到输出的某一张图
		targets += img_idx * filter_channel * out_pixs + filt_idx * out_pixs \
				   + out_row * out_width + out_col;

		float out_value = 0;

		for(int k = 0; k < in_channel; k++){
			const float *x_offset = x + k*in_pixs;
			const float *w_offset = w + k*filt_pixs;

			for(int i = 0; i < filter_height; i++){
				int in_row = out_row * stride_height + i;
				const float *x_offset_1 = x_offset + in_row*in_width;
				const float *w_offset_1 = w_offset + i*filter_width;

				for(int j = 0; j < filter_width; j++){
					int in_col = out_col * stride_width + j;

					if(in_row < in_height && in_col < in_width){
						out_value += x_offset_1[in_col] \
							 *w_offset_1[j];
					}
				}
			}
		}
		__syncthreads();
		targets[0] = out_value + bias[0];
	}
}

__global__ void backward_convolution(const float* dE_dy, const float *w, \
		float* targets, \
		const int box_in_height, const int box_in_width, \
		const int box_out_height, const int box_out_width, \
		const int out_channel, const int in_channel, \
		const int out_height, const int out_width, \
		const int filter_height, const int filter_width, \
		const int stride_height, const int stride_width, \
		const int box_num_height, const int box_num_width){

	extern __shared__ float result[];

	const int num_box = box_num_height * box_num_width;	
	const int img_idx = blockIdx.x;
	const int in_channel_idx = blockIdx.y / num_box;
	const int box_idx = blockIdx.y % num_box; 
	const int box_row_idx = box_idx / box_num_width;
	const int box_col_idx = box_idx % box_num_width;

	///反向求的输入width
	const int in_height = box_in_height * box_num_height; 
	const int in_width = box_in_width * box_num_width; 
	int in_pixs = in_height * in_width;
	int filt_pixs = filter_height * filter_width;
	int out_pixs = out_height * out_width;

	//输出的行列idx,当pooling之后的width小于32时，box_row_idx会等于0
	int out_row = box_out_height * box_row_idx + threadIdx.y;
	int out_col = box_out_width * box_col_idx + threadIdx.x;

	if(out_row < out_height && out_col < out_width){
		targets += img_idx * in_channel * in_pixs + in_channel_idx * in_pixs \
				   + box_row_idx * box_in_height * in_width \
				   + box_col_idx * box_in_width;
		dE_dy += img_idx*out_channel*out_pixs + out_row*out_width + out_col;
		w += in_channel_idx*filt_pixs;

		int interval_height = out_height - box_out_height * (box_num_height - 1);
		int interval_width = out_width - box_out_width* (box_num_width - 1);
		int tmp_row = threadIdx.y;
		int tmp_col = threadIdx.x;
		while(tmp_row < box_in_height){
			tmp_col = threadIdx.x;
			while(tmp_col < box_in_width){
				result[tmp_row*box_in_width + tmp_col] = 0;
				tmp_col += interval_width;
			}
			tmp_row += interval_height;
		}

		float ele = 0;
		for(int k = 0; k < out_channel; k++){
			const float *dE_dy_offset = dE_dy + k*out_pixs;
			const float *w_offset = w + k*filt_pixs*in_channel;
			for(int i = 0; i < filter_height; i++){
				int box_in_row = threadIdx.y*stride_height + i;
				const float *w_offset_1 = w_offset + i*filter_width;
				float *result_offset = result + box_in_row*box_in_width;

				for(int j = 0; j < filter_width; j++){
					int box_in_col = threadIdx.x*stride_width + j;

					if(box_in_row < box_in_height \
							&& box_in_col < box_in_width){
						ele = dE_dy_offset[0]*w_offset_1[j];
						atomicAdd(result_offset+box_in_col, ele);
						__syncthreads();
					}
				}
			}
		}

		tmp_row = threadIdx.y;
		while(tmp_row < box_in_height){
			tmp_col = threadIdx.x;
			float *target_offset = targets + tmp_row*in_width;
			float *result_offset = result + tmp_row*box_in_width;

			while(tmp_col < box_in_width){
				target_offset[tmp_col] = result_offset[tmp_col];
				tmp_col += interval_width;
			}
			tmp_row += interval_height;
		}
	}
}


__global__ void compute_convolution_derivs(const float* dE_dy, const float *x, \
		float* dE_dw, const int box_out_height, const int box_out_width, \
		const int out_channel, const int in_channel, const int in_height, \
		const int in_width, const int out_height, const int out_width, \
		const int filter_height, const int filter_width, \
		const int stride_height, const int stride_width, \
		const int box_num_height, const int box_num_width){

	extern __shared__ float result[];

	int in_pixs = in_height * in_width;
	int filt_pixs = filter_height * filter_width;
	int out_pixs = out_height * out_width;
	int box_out_pixs = box_out_height * box_out_width;

	const int num_box = box_num_height * box_num_width;	

	const int img_idx = blockIdx.x / out_channel;
	const int out_channel_idx = blockIdx.x % out_channel;

	const int in_channel_idx = blockIdx.y / (num_box*filt_pixs);

	int tmp = blockIdx.y%(num_box*filt_pixs);
	const int box_idx = tmp / filt_pixs;
	const int box_row_idx = box_idx / box_num_width;
	const int box_col_idx = box_idx % box_num_width;
	const int filt_row_idx = (tmp % filt_pixs) / filter_width; 
	const int filt_col_idx = (tmp % filt_pixs) % filter_width; 

	int out_row = box_out_height * box_row_idx + threadIdx.y;
	int out_col = box_out_width * box_col_idx + threadIdx.x;
	int in_row = out_row*stride_height + filt_row_idx;
	int in_col = out_col*stride_width + filt_col_idx;

	if(threadIdx.y < box_out_height && threadIdx.x < box_out_width)
		result[threadIdx.y*box_out_width+threadIdx.x] = 0;

	if(out_row < out_height && out_col < out_width \
			&& in_row < in_height && in_col < in_width){
		x += img_idx*in_channel*in_pixs + in_channel_idx*in_pixs \
			 + in_row*in_width + in_col; 
		dE_dw += img_idx*out_channel*in_channel*filt_pixs*num_box \
				 + out_channel_idx*in_channel*filt_pixs*num_box \
				 + in_channel_idx*filt_pixs*num_box + box_idx*filt_pixs \
				 + filt_row_idx*filter_width + filt_col_idx;
		dE_dy += img_idx*out_channel*out_pixs + out_channel_idx*out_pixs \
				 + out_row*out_width + out_col;

		int idx = threadIdx.y*box_out_width + threadIdx.x;
		
		result[idx] = dE_dy[0]*x[0];

		int pow2Length = box_out_pixs;
		if(pow2Length & (pow2Length - 1)){
			while(pow2Length & (pow2Length - 1)){
				pow2Length &= pow2Length - 1;
			}
		}
		__syncthreads();

		if(idx >= pow2Length && idx < box_out_pixs)
			result[idx - pow2Length] += result[idx];
		__syncthreads();

		for(int activeThreads = (pow2Length >> 1); activeThreads; \
				activeThreads >>= 1){
			if(idx < activeThreads)
				result[idx] += result[idx+activeThreads];
			__syncthreads();
		}
		if(idx == 0)
			dE_dw[0] = result[0];
	}
}

__global__ void compact_dervis_w(const float* unranged_dE_dw, \
		float* dE_dw, const int filter_height, const int filter_width, \
		const int box_num_height, const int box_num_width, \
		const int minibatch_size, const int in_channel, const int out_channel){
	//此处grid是一维
	int filt_idx = blockIdx.y;	
	const int num_box = box_num_height * box_num_width;	
	int filt_pixs = filter_height * filter_width;
	if(threadIdx.y < filter_height && threadIdx.x < filter_width){
		dE_dw += filt_idx*filt_pixs + threadIdx.y*filter_width + threadIdx.x;
		unranged_dE_dw += filt_idx*filt_pixs*num_box \
						  + threadIdx.y*filter_width + threadIdx.x;
		dE_dw[0] = 0;
		int value = in_channel*out_channel*filt_pixs*num_box;
		for(int k = 0; k < minibatch_size; k++){
			const float *unranged_dE_dw_offset = unranged_dE_dw + k*value;
			for(int i = 0; i < num_box; i++){
				dE_dw[0] += unranged_dE_dw_offset[i*filt_pixs]; 
			}

		}
	}
}

__global__ void compute_derivs_of_bias(const float* dE_dy, float* targets, \
		const int out_height, const int out_width, const int out_channel, \
		const int box_out_height, const int box_out_width, \
		const int box_num_height, const int box_num_width){

	const int num_box = box_num_height * box_num_width;	
	const int img_idx = blockIdx.x;
	const int filt_idx = blockIdx.y / num_box;
	const int box_idx = blockIdx.y % num_box; 
	const int box_row_idx = box_idx / box_num_width;
	const int box_col_idx = box_idx % box_num_width;

	int out_pixs = out_height * out_width;
	int box_out_pixs = box_out_height * box_out_width;

	extern __shared__ float result[];

	//输出的行列idx，当输出大于MAX_THREAD_SIZE的时候每个线程都做了计算
	int out_row = box_out_height * box_row_idx + threadIdx.y;
	int out_col = box_out_width * box_col_idx + threadIdx.x;

	if(threadIdx.y < box_out_height && threadIdx.x < box_out_width)
		result[threadIdx.y*box_out_width+threadIdx.x] = 0;

	if(out_row < out_height && out_col < out_width){
		dE_dy += img_idx*out_pixs*out_channel + filt_idx*out_pixs \
				 + out_row*out_width + out_col;
		
		targets += img_idx*out_channel*num_box+filt_idx*num_box \
				   +box_row_idx*box_num_width + box_col_idx;

		int idx = threadIdx.y*box_out_width + threadIdx.x;
		result[idx] = dE_dy[0];

		int pow2Length = box_out_pixs;
		if(pow2Length & (pow2Length - 1)){
			while(pow2Length & (pow2Length - 1)){
				pow2Length &= pow2Length - 1;
			}
		}
		__syncthreads();

		if(idx >= pow2Length && idx < box_out_pixs)
			result[idx - pow2Length] += result[idx];
		__syncthreads();

		for(int activeThreads = (pow2Length >> 1); activeThreads; \
				activeThreads >>= 1){
			if(idx < activeThreads)
				result[idx] += result[idx+activeThreads];
			__syncthreads();
		}

		if(idx == 0)
			targets[0] = result[0];
	}
}


__global__ void pad_to_ori(float* dst, const float* src, const int num_kernel, \
		const int img_height, const int img_width, \
		const int padded_img_height, const int padded_img_width, \
		const int img_channel){

	int index;
	int img_pixs = img_height * img_width;
	int padded_img_pixs = padded_img_height * padded_img_width;
	int pad_height = (padded_img_height - img_height) / 2;
	int pad_width = (padded_img_width - img_width) / 2;
	int one_img_len = img_channel*img_pixs;
	CUDA_KERNEL_LOOP(idx, num_kernel){

		const int img_idx = idx / one_img_len;
		const int src_col = idx % one_img_len;
		const int img_channel_idx = src_col / img_pixs;
		const int img_row = (src_col % img_pixs) / img_width; 
		const int img_col = (src_col % img_pixs) % img_width; 
		index = img_idx * img_channel * padded_img_pixs \
				+ img_channel_idx * padded_img_pixs \
				+ (img_row + pad_height) * padded_img_width \
				+ (img_col + pad_width); 
		dst[idx] = src[index]; 
	}
}

__global__ void ori_to_padding(const float* src, float* dst, const int num_kernel, \
		const int img_height, const int img_width, const int padded_img_height, \
		const int padded_img_width, const int img_channel){

	int index;
	int img_pixs = img_height * img_width;
	int padded_img_pixs = padded_img_height * padded_img_width;
	int pad_height = (padded_img_height - img_height) / 2;
	int pad_width = (padded_img_width - img_width) / 2;
	int one_img_len = img_channel*img_pixs;
	CUDA_KERNEL_LOOP(idx, num_kernel){

		const int img_idx = idx / one_img_len;
		const int src_col = idx % one_img_len;
		const int img_channel_idx = src_col / img_pixs;
		const int img_row = (src_col % img_pixs) / img_width; 
		const int img_col = (src_col % img_pixs) % img_width; 
		index = img_idx * img_channel * padded_img_pixs \
				+ img_channel_idx * padded_img_pixs \
				+ (img_row + pad_height) * padded_img_width \
				+ (img_col + pad_width); 
		dst[index] = src[idx]; 
	}

}

__global__ void max_pooling(const float* x, float* targets, int* maxPoolPos, \
		const int in_height, const int in_width, \
		const int in_channels, const int out_height, const int out_width, \
		const int filter_height, const int filter_width, \
		const int stride_height, const int stride_width,  \
		const int box_out_height, const int box_out_width, \
		const int box_num_height, const int box_num_width){

	const int num_box = box_num_height * box_num_width;	
	const int img_idx = blockIdx.x;
	const int filt_idx = blockIdx.y / num_box;
	const int box_idx = blockIdx.y % num_box; 
	const int box_row_idx = box_idx / box_num_width;
	const int box_col_idx = box_idx % box_num_width;

	int conv_pixs = in_height * in_width;
	int pool_pixs = out_height * out_width;

	//输出的行列idx，当输出大于MAX_THREAD_SIZE的时候每个线程都做了计算
	int out_row = MAX_THREAD_SIZE * box_row_idx + threadIdx.y;
	int out_col = MAX_THREAD_SIZE * box_col_idx + threadIdx.x;

	if(out_row < out_height && out_col < out_width){
		//首先定位哪一个batch的哪一个channel的哪一张图，然后寻找在这张图上位置
		x += img_idx * in_channels * conv_pixs + filt_idx * conv_pixs;
		targets += img_idx * in_channels * pool_pixs + filt_idx * pool_pixs \
				   + out_row * out_width + out_col;
		maxPoolPos += img_idx * in_channels * pool_pixs + filt_idx * pool_pixs \
					  + out_row * out_width + out_col;

		float max_value = x[out_row*stride_height*in_width+out_col*stride_width];
		int max_pos = 0;
		for(int i = 0; i < filter_height; i++){
			int conv_row = out_row * stride_height + i;
			const float* x_offset = x + conv_row*in_width; 

			for(int j = 0; j < filter_width; j++){
				int conv_col = out_col * stride_width + j;

				if(conv_row < in_height && conv_col < in_width){
					if(x_offset[conv_col]>max_value){
						max_value = x_offset[conv_col];
						max_pos = i*filter_width +j;
					}
				}
			}
		}
		__syncthreads();

		targets[0] = max_value;
		maxPoolPos[0] = max_pos;
	}
}

__global__ void avg_pooling(const float* x, float* targets, \
		const int in_height, const int in_width, \
		const int in_channels, const int out_height, const int out_width, \
		const int filter_height, const int filter_width, \
		const int stride_height, const int stride_width,  \
		const int box_out_height, const int box_out_width, \
		const int box_num_height, const int box_num_width){

	const int num_box = box_num_height * box_num_width;	
	const int img_idx = blockIdx.x;
	const int filt_idx = blockIdx.y / num_box;
	const int box_idx = blockIdx.y % num_box; 
	const int box_row_idx = box_idx / box_num_width;
	const int box_col_idx = box_idx % box_num_width;

	int conv_pixs = in_height * in_width;
	int pool_pixs = out_height * out_width;

	int out_row = MAX_THREAD_SIZE * box_row_idx + threadIdx.y;
	int out_col = MAX_THREAD_SIZE * box_col_idx + threadIdx.x;

	if(out_row < out_height && out_col < out_width){
		x += img_idx * in_channels * conv_pixs + filt_idx * conv_pixs; 
		targets += img_idx * in_channels * pool_pixs + filt_idx * pool_pixs \
				   + out_row * out_width + out_col;

		float avg_value = 0;
		for(int i = 0; i < filter_height; i++){
			int conv_row = out_row * stride_height + i;
			const float* x_offset = x + conv_row*in_width; 
			
			for(int j = 0; j < filter_width; j++){
				int conv_col = out_col * stride_width + j;
				if(conv_row < in_height && conv_col < in_width){
					avg_value += x_offset[conv_col];
				}
			}
		}
		__syncthreads();

		targets[0] = avg_value / (filter_height * filter_width);
	}
}



__global__ void compute_dE_dy_max(const float* dE_dy_i, float* targets, \
		int* maxPoolPos, \
		const int box_in_height, const int box_in_width, \
		const int box_out_height, const int box_out_width, \
		const int num_filters, \
		const int out_height, const int out_width, \
		const int filter_height, const int filter_width, \
		const int stride_height, const int stride_width,  \
		const int box_num_height, const int box_num_width){

	const int num_box = box_num_height * box_num_width;	
	const int img_idx = blockIdx.x;
	const int filt_idx = blockIdx.y / num_box;
	const int box_idx = blockIdx.y % num_box; 
	const int box_row_idx = box_idx / box_num_width;
	const int box_col_idx = box_idx % box_num_width;

	const int in_height = box_in_height * box_num_height; 
	const int in_width = box_in_width * box_num_width; 
	int in_pixs = in_height * in_width;
	int pool_pixs = out_height * out_width;

	//输出的行列idx
	int out_row = box_out_height * box_row_idx + threadIdx.y;
	int out_col = box_out_width * box_col_idx + threadIdx.x;

	int in_row = box_in_height * box_row_idx + threadIdx.y * stride_height;
	int	in_col = box_in_width * box_col_idx + threadIdx.x * stride_width;

	//共享内存的大小只有计算一块pool输出的对应conv输入
	extern __shared__ float result[];

	if(out_row < out_height && out_col < out_width){
		targets += img_idx * num_filters * in_pixs + filt_idx * in_pixs \
				   + in_row * in_width + in_col; 
		dE_dy_i += img_idx * num_filters * pool_pixs + filt_idx * pool_pixs \
				   + out_row * out_width + out_col;
		maxPoolPos += img_idx * num_filters * pool_pixs + filt_idx * pool_pixs \
					  + out_row * out_width + out_col;

		int posIdx= threadIdx.y * box_in_width * stride_height \
					+ threadIdx.x * stride_width;

		int interval_height = out_height - box_out_height * (box_num_height - 1);
		int interval_width = out_width - box_out_width* (box_num_width - 1);
		int tmp_row = threadIdx.y;
		int tmp_col = threadIdx.x;
		while(tmp_row < box_in_height){
			tmp_col = threadIdx.x;
			float* result_offset = result + tmp_row*box_in_width;

			while(tmp_col < box_in_width){
				result_offset[tmp_col] = 0;
				tmp_col += interval_width;
			}
			tmp_row += interval_height;
		}

		int pos = maxPoolPos[0];
		int row = pos / filter_width;
		int col = pos % filter_width;

		posIdx += row * box_in_width + col;

		float ele = dE_dy_i[0];

		atomicAdd(result+posIdx, ele);
		__syncthreads();

		if(in_row + row < in_height && in_col + col < in_width){
			targets[row * in_height + col] = result[posIdx];
		}
	}
}

__global__ void compute_dE_dy_avg(const float* dE_dy_i, float* targets, \
		const int box_in_height, const int box_in_width, \
		const int box_out_height, const int box_out_width, \
		const int num_filters, \
		const int out_height, const int out_width, \
		const int filter_height, const int filter_width, \
		const int stride_height, const int stride_width,  \
		const int box_num_height, const int box_num_width){

	//这里的out代表的是pooling的输出
	extern __shared__ float result[];

	const int num_box = box_num_height * box_num_width;	
	const int img_idx = blockIdx.x;
	const int filt_idx = blockIdx.y / num_box;
	const int box_idx = blockIdx.y % num_box; 
	const int box_row_idx = box_idx / box_num_width;
	const int box_col_idx = box_idx % box_num_width;

	///反向求的输入width
	const int in_height = box_in_height * box_num_height; 
	const int in_width = box_in_width * box_num_width; 
	int in_pixs = in_height * in_width;
	int pool_pixs = out_height * out_width;

	//输出的行列idx,当pooling之后的width小于32时，box_row_idx会等于0
	int out_row = box_out_height * box_row_idx + threadIdx.y;
	int out_col = box_out_width * box_col_idx + threadIdx.x;

	if(out_row < out_height && out_col < out_width){
		//计算本块pooling对应的输入conv块起始位置
		//本线程对应的pooling值
		targets += img_idx * num_filters * in_pixs + filt_idx * in_pixs \
				   + box_row_idx * box_in_height * in_width \
				   + box_col_idx * box_in_width;
		dE_dy_i += img_idx * num_filters * pool_pixs + filt_idx * pool_pixs \
				   + out_row * out_width + out_col;

		float ele;
		int interval_height = out_height - box_out_height * (box_num_height - 1);
		int interval_width = out_width - box_out_width* (box_num_width - 1);
		int tmp_row = threadIdx.y;
		int tmp_col = threadIdx.x;
		while(tmp_row < box_in_height){
			tmp_col = threadIdx.x;
			float* result_offset = result + tmp_row*box_in_width;

			while(tmp_col < box_in_width){
				result_offset[tmp_col] = 0;
				tmp_col += interval_width;
			}
			tmp_row += interval_height;
		}

		int filt_pixs = filter_height*filter_width;
		for(int i = 0; i < filter_height; i++){
			int box_in_row = threadIdx.y * stride_height + i;
			float *result_offset = result + box_in_row*box_in_width;
			for(int j = 0; j < filter_width; j++){
				int box_in_col = threadIdx.x * stride_width + j;
				if(box_in_row < box_in_height && box_in_col < box_in_width){
					ele = dE_dy_i[0] / filt_pixs;
					atomicAdd(result_offset+box_in_col, ele);
					__syncthreads();
				}
			}
		}
		tmp_row = threadIdx.y;
		tmp_col = threadIdx.x;
		while(tmp_row < box_in_height){
			tmp_col = threadIdx.x;
			float* result_offset = result + tmp_row*box_in_width;
			float* target_offset = targets + tmp_row*in_width;
			while(tmp_col < box_in_width){
				target_offset[tmp_col] = result_offset[tmp_col];
				tmp_col += interval_width;
			}
			tmp_row += interval_height;
		}
	}

}

__global__ void compute_dE_dy(const float* y_j, const int* labels, \
		float* dE_dy_j, const int width) {
	const int tx = blockIdx.x;
	const int ty = blockIdx.x * width + threadIdx.x;

	const int lab = labels[tx];

	if(threadIdx.x < width)
		dE_dy_j[ty] = y_j[ty] - (lab == threadIdx.x);
	__syncthreads();
}


__global__ void compactOverlap(float* src, float* targets, \
		const int in_height, const int in_width, const int in_channel, \
		const int overlap_height, const int overlap_width, \
		const int box_in_height, const int box_in_width, \
		const int box_num_height, const int box_num_width){

	const int img_idx = blockIdx.x;
	const int filt_idx = threadIdx.x;

	const int unfold_in_height = box_in_height * box_num_height; 
	const int unfold_in_width = box_in_width * box_num_width; 
	const int in_pixs = in_height * in_width;
	const int unfold_in_pix = unfold_in_height*unfold_in_width;

	src += img_idx*in_channel*unfold_in_pix + filt_idx*unfold_in_pix;
	targets += img_idx*in_channel*in_pixs + filt_idx*in_pixs;
 
	for(int i = 0; i < unfold_in_height; i++){
		int in_row = i - overlap_height*(i/box_in_height);
		float *target_offset = targets + in_row*in_width;
		float *src_offset = src + i*unfold_in_width;

		for(int j = 0; j < unfold_in_width; j++){
			int in_col = j - overlap_width*(j/box_in_width);
			if(in_row >= in_height || in_col >= in_width)
				break;
			target_offset[in_col] += src_offset[j];
		}
	}
}

















