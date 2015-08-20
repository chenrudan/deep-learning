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
		const float* bias, \
		float* targets, const int in_size, const int in_channel, \
		const int out_size, const int filter_size, const int filter_channel, \
		const int stride, const int box_out_size, const int box_num_size){

	const int num_box = box_num_size * box_num_size;	
	const int img_idx = blockIdx.x;
	const int filt_idx = blockIdx.y / num_box;
	const int box_idx = blockIdx.y % num_box; 
	const int box_row_idx = box_idx / box_num_size;
	const int box_col_idx = box_idx % box_num_size;


	int in_pixs = in_size * in_size;
	int out_pixs = out_size * out_size;
	int filt_pixs = filter_size * filter_size;

	//输出的行列idx，当输出大于MAX_THREAD_SIZE的时候每个线程都做了计算
	int out_row = MAX_THREAD_SIZE * box_row_idx + threadIdx.y;
	int out_col = MAX_THREAD_SIZE * box_col_idx + threadIdx.x;

	if(out_row < out_size && out_col < out_size){
		//x定位哪一个batch
		x += img_idx * in_channel * in_pixs;
		//定位到哪一个filter
		w += filt_idx * in_channel* filt_pixs;
		bias += filt_idx;
		//输出定位到输出的某一张图
		targets += img_idx * filter_channel * out_pixs + filt_idx * out_pixs \
				   + out_row * out_size + out_col;

		float out_value = 0;

		for(int k = 0; k < in_channel; k++){
			//将需要用到的输入一个channel值保存到共享内存，每次计算一个channel的结果
			for(int i = 0; i < filter_size; i++){
				for(int j = 0; j < filter_size; j++){
					int in_row = out_row * stride + i;
					int in_col = out_col * stride + j;
					if(in_row < in_size && in_col < in_size){
						out_value += x[k*in_pixs + in_row*in_size + in_col] \
							 *w[k*filt_pixs+i*filter_size+j];
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
		const int box_in_size, const int box_out_size, \
		const int out_channel, const int in_channel, \
		const int out_size, const int filter_size, \
		const int stride, const int box_num_size){

	extern __shared__ float result[];

	const int num_box = box_num_size * box_num_size;	
	const int img_idx = blockIdx.x;
	const int in_channel_idx = blockIdx.y / num_box;
	const int box_idx = blockIdx.y % num_box; 
	const int box_row_idx = box_idx / box_num_size;
	const int box_col_idx = box_idx % box_num_size;

	///反向求的输入width
	const int in_size = box_in_size * box_num_size; 
	int in_pixs = in_size * in_size;
	int filt_pixs = filter_size * filter_size;
	int out_pixs = out_size * out_size;

	//输出的行列idx,当pooling之后的width小于32时，box_row_idx会等于0
	int out_row = box_out_size * box_row_idx + threadIdx.y;
	int out_col = box_out_size * box_col_idx + threadIdx.x;

	if(out_row < out_size && out_col < out_size){
		targets += img_idx * in_channel * in_pixs + in_channel_idx * in_pixs \
				   + box_row_idx * box_in_size * in_size \
				   + box_col_idx * box_in_size;
		dE_dy += img_idx * out_channel * out_pixs;

		float ele;
		int interval = out_size - box_out_size * (box_num_size - 1);
		for(int i = threadIdx.x; i < box_in_size; i+=interval)
			for(int j = threadIdx.y; j < box_in_size; j+=interval)
				result[i*box_in_size + j] = 0;

		for(int k = 0; k < out_channel; k++){
			for(int i = 0; i < filter_size; i++){
				for(int j = 0; j < filter_size; j++){
					int box_in_row = threadIdx.y*stride + i;
					int box_in_col = threadIdx.x*stride + j;
					if(box_in_row < box_in_size && box_in_col < box_in_size){
						ele = dE_dy[k*out_pixs+(i+out_row)*out_size+j+out_col] \
							  *w[k*filt_pixs*in_channel+i*filter_size+j];
						__syncthreads();
						atomicAdd(result + box_in_row*box_in_size+box_in_col, ele);
						__syncthreads();
					}
				}
			}
		}
		for(int i = threadIdx.y; i < box_in_size; i+=interval)
			for(int j = threadIdx.x; j < box_in_size; j+=interval)
				targets[i * in_size + j] = result[i * box_in_size + j];
		}
}


__global__ void compute_convolution_derivs(const float* dE_dy, const float *x, \
		float* dE_dw, const int box_in_size, const int box_out_size, \
		const int out_channel, const int in_channel, \
		const int out_size, const int filter_size, \
		const int stride, const int box_num_size){

	extern __shared__ float result[];

	const int num_box = box_num_size * box_num_size;	
	const int img_idx = blockIdx.x;
	const int out_channel_idx = blockIdx.y % out_channel;
	const int in_channel_idx = (blockIdx.y/out_channel) / num_box;
	const int box_idx = (blockIdx.y/out_channel) % num_box; 
	const int box_row_idx = box_idx / box_num_size;
	const int box_col_idx = box_idx % box_num_size;

	///反向求的输入width
	const int in_size = box_in_size * box_num_size; 
	int in_pixs = in_size * in_size;
	int filt_pixs = filter_size * filter_size;
	int out_pixs = out_size * out_size;

	int out_row = box_out_size * box_row_idx + threadIdx.y;
	int out_col = box_out_size * box_col_idx + threadIdx.x;

	
	if(out_row < out_size && out_col < out_size){
		x += img_idx*in_channel*in_pixs + in_channel_idx*in_pixs;
		dE_dw += out_channel_idx*in_channel*filt_pixs*num_box \
				 + in_channel_idx*filt_pixs*num_box + box_idx*filt_pixs;
		dE_dy += img_idx*out_channel*out_pixs + out_channel_idx*out_pixs;

		int tmp_row = threadIdx.y;
		int tmp_col = threadIdx.x;
		while(tmp_row < filter_size && tmp_col < filter_size){
			result[tmp_row*filter_size + tmp_col] = 0;
			tmp_row += blockDim.y;
			tmp_col += blockDim.x;
		}

		float ele;
		for(int i = 0; i < filter_size; i++){
			for(int j = 0; j < filter_size; j++){
				int in_row = out_row*stride + i;
				int in_col = out_col*stride + j;
				if(in_row < in_size && in_col < in_size){
					ele = dE_dy[(i+out_row)*out_size+j+out_col] \
							  *x[in_row*in_size + in_col];
					__syncthreads();
					atomicAdd(result+i*filter_size+j, ele);
					__syncthreads();
				}
			}
		}
		tmp_row = threadIdx.y;
		tmp_col = threadIdx.x;
		while(tmp_row < filter_size && tmp_col < filter_size){
			dE_dw[tmp_row*filter_size+tmp_col] \
					= result[tmp_row*filter_size+tmp_col];
			tmp_row += blockDim.y;
			tmp_col += blockDim.x;
		}
		__syncthreads();
	}
}

__global__ void compact_dervis_w(const float* unranged_dE_dw, \
		float* dE_dw, const int filter_size, const int box_num_size){
	//此处grid是一维
	int filt_idx = blockIdx.x;	
	const int num_box = box_num_size * box_num_size;	
	int filt_pixs = filter_size * filter_size;
	if(threadIdx.y < filter_size && threadIdx.x < filter_size){
		dE_dw += filt_idx*filt_pixs + threadIdx.y*filter_size + threadIdx.x;
		unranged_dE_dw += filt_idx*filt_pixs*num_box \
						  + threadIdx.y*filter_size + threadIdx.x;
		dE_dw[0] = 0;
		for(int i = 0; i < num_box; i++){
			dE_dw[0] += unranged_dE_dw[i*filt_pixs]; 
		}
	}
}















__global__ void ori_to_padding(const float* src, float* dst, const int num_kernel, \
		const int img_size, const int padded_img_size, const int img_channel){

	int index;
	int img_pixs = img_size * img_size;
	int padded_img_pixs = padded_img_size * padded_img_size;
	int pad = (padded_img_size - img_size) / 2;
	CUDA_KERNEL_LOOP(idx, num_kernel){

		const int img_idx = idx / (img_channel * img_pixs);
		const int src_col = idx % (img_channel * img_pixs);
		const int img_channel_idx = src_col / img_pixs;
		const int img_row = (src_col % img_pixs) / img_size; 
		const int img_col = (src_col % img_pixs) % img_size; 
		index = img_idx * img_channel * padded_img_pixs + img_channel_idx * padded_img_pixs \
				+ (img_row + pad) * padded_img_size \
				+ (img_col + pad); 
		dst[index] = src[idx]; 
	}

}

__global__ void reshape_dE_dy2(float* un_dE_dy, const float* dE_dy, \
		const int num_kernel, const int conv_forward_size, \
		const int filter_channel){
	int index;
	CUDA_KERNEL_LOOP(idx, num_kernel){
		int conv_pixs = conv_forward_size * conv_forward_size;

		const int dst_row = idx / (conv_pixs * filter_channel);
		const int dst_col = idx % (conv_pixs * filter_channel);
		const int ori_col = dst_col % conv_pixs;
		const int ori_row = dst_row * filter_channel + dst_col / conv_pixs;
		index = ori_row * conv_pixs + ori_col;
		un_dE_dy[index] = dE_dy[idx]; 
	}
}

__global__ void reshape_dE_db_tmp(float* dst, const float* ori, \
		const int num_kernel, const int filter_channel){
	int index;
	CUDA_KERNEL_LOOP(idx, num_kernel){

		const int ori_col = idx % filter_channel;
		const int ori_row = idx / filter_channel;
		index = ori_row * filter_channel + ori_col;
		dst[index] = ori[idx]; 
	}
}

__global__ void max_pooling(const float* convOutputs, float* targets, int* maxPoolPos, \
		const int conv_forward_size, const int in_channels, const int pool_forward_size, \
		const int max_pool_size, const int stride, \
		const int box_out_size, const int box_num_size){

	const int num_box = box_num_size * box_num_size;	
	const int img_idx = blockIdx.x;
	const int filt_idx = blockIdx.y / num_box;
	const int box_idx = blockIdx.y % num_box; 
	const int box_row_idx = box_idx / box_num_size;
	const int box_col_idx = box_idx % box_num_size;

	int conv_pixs = conv_forward_size * conv_forward_size;
	int pool_pixs = pool_forward_size * pool_forward_size;

	//输出的行列idx，当输出大于MAX_THREAD_SIZE的时候每个线程都做了计算
	int out_row = MAX_THREAD_SIZE * box_row_idx + threadIdx.y;
	int out_col = MAX_THREAD_SIZE * box_col_idx + threadIdx.x;

	if(out_row < pool_forward_size && out_col < pool_forward_size){
		//首先定位哪一个batch的哪一个channel的哪一张图，然后寻找在这张图上位置
		convOutputs += img_idx * in_channels * conv_pixs + filt_idx * conv_pixs;
		targets += img_idx * in_channels * pool_pixs + filt_idx * pool_pixs \
				   + out_row * pool_forward_size + out_col;
		maxPoolPos += img_idx * in_channels * pool_pixs + filt_idx * pool_pixs \
					  + out_row * pool_forward_size + out_col;

		float max_value = convOutputs[out_row*stride*conv_forward_size+out_col*stride];
		int max_pos = 0;
		for(int i = 0; i < max_pool_size; i++){
			for(int j = 0; j < max_pool_size; j++){
				int conv_row = out_row * stride + i;
				int conv_col = out_col * stride + j;
				if(conv_row < conv_forward_size && conv_col < conv_forward_size){
					if(convOutputs[conv_row*conv_forward_size+conv_col]>max_value){
						max_value = convOutputs[conv_row*conv_forward_size \
									+ conv_col];
						max_pos = i*max_pool_size +j;
					}
				}
			}
		}
		__syncthreads();

		targets[0] = max_value;
		maxPoolPos[0] = max_pos;
	}
}

__global__ void avg_pooling(const float* convOutputs, float* targets, \
		const int conv_forward_size, const int in_channels, \
		const int pool_forward_size, const int avg_pool_size, \
		const int stride, const int box_out_size, const int box_num_size){

	const int num_box = box_num_size * box_num_size;	
	const int img_idx = blockIdx.x;
	const int filt_idx = blockIdx.y / num_box;
	const int box_idx = blockIdx.y % num_box; 
	const int box_row_idx = box_idx / box_num_size;
	const int box_col_idx = box_idx % box_num_size;


	int conv_pixs = conv_forward_size * conv_forward_size;
	int pool_pixs = pool_forward_size * pool_forward_size;

	int out_row = MAX_THREAD_SIZE * box_row_idx + threadIdx.y;
	int out_col = MAX_THREAD_SIZE * box_col_idx + threadIdx.x;

	if(out_row < pool_forward_size && out_col < pool_forward_size){
		convOutputs += img_idx * in_channels * conv_pixs + filt_idx * conv_pixs; 
		targets += img_idx * in_channels * pool_pixs + filt_idx * pool_pixs \
				   + out_row * pool_forward_size + out_col;

		float avg_value = 0;
		for(int i = 0; i < avg_pool_size; i++){
			for(int j = 0; j < avg_pool_size; j++){
				int conv_row = out_row * stride + i;
				int conv_col = out_col * stride + j;
				if(conv_row < conv_forward_size && conv_col < conv_forward_size){
					avg_value += convOutputs[conv_row*conv_forward_size+conv_col];
				}
			}
		}
		__syncthreads();

		targets[0] = avg_value / (avg_pool_size * avg_pool_size);
	}
}



__global__ void compute_dE_dy_max(const float* dE_dy_i, float* targets, \
		int* maxPoolPos, const int box_in_size, const int box_out_size, \
		const int num_filters, \
		const int pool_forward_size, const int max_pool_size, \
		const int stride, const int box_num_size){

	const int num_box = box_num_size * box_num_size;	
	const int img_idx = blockIdx.x;
	const int filt_idx = blockIdx.y / num_box;
	const int box_idx = blockIdx.y % num_box; 
	const int box_row_idx = box_idx / box_num_size;
	const int box_col_idx = box_idx % box_num_size;

	const int in_size = box_in_size * box_num_size; 
	int in_pixs = in_size * in_size;
	int pool_pixs = pool_forward_size * pool_forward_size;

	//输出的行列idx
	int out_row = box_out_size * box_row_idx + threadIdx.y;
	int out_col = box_out_size * box_col_idx + threadIdx.x;

	int in_row = box_in_size * box_row_idx + threadIdx.y * stride;
	int	in_col = box_in_size * box_col_idx + threadIdx.x * stride;

	//共享内存的大小只有计算一块pool输出的对应conv输入
	extern __shared__ float result[];

	if(out_row < pool_forward_size && out_col < pool_forward_size){
		targets += img_idx * num_filters * in_pixs + filt_idx * in_pixs \
				   + in_row * in_size + in_col; 
		dE_dy_i += img_idx * num_filters * pool_pixs + filt_idx * pool_pixs \
				   + out_row * pool_forward_size + out_col;
		maxPoolPos += img_idx * num_filters * pool_pixs + filt_idx * pool_pixs \
					  + out_row * pool_forward_size + out_col;

		int posIdx= threadIdx.y * box_in_size * stride \
					+ threadIdx.x * stride;

		int interval = pool_forward_size - box_out_size * (box_num_size - 1);
		for(int i = threadIdx.y; i < box_in_size; i += interval)
			for(int j = threadIdx.x; j < box_in_size; j += interval)
				result[i*box_in_size + j] = 0;

		int pos = maxPoolPos[0];
		int row = pos / max_pool_size;
		int col = pos % max_pool_size;

		posIdx += row * box_in_size + col;

		float ele = dE_dy_i[0];

		__syncthreads();
		atomicAdd(result+posIdx, ele);
		__syncthreads();

		if(in_row + row < in_size && in_col + col < in_size){
			targets[row * in_size + col] = result[posIdx];
		}
	}
}

__global__ void compute_dE_dy_avg(const float* dE_dy_i, float* targets, \
		const int box_in_size, const int box_out_size, \
		const int num_filters, \
		const int pool_forward_size, const int avg_pool_size, \
		const int stride, const int box_num_size){

	//这里的out代表的是pooling的输出
	extern __shared__ float result[];

	const int num_box = box_num_size * box_num_size;	
	const int img_idx = blockIdx.x;
	const int filt_idx = blockIdx.y / num_box;
	const int box_idx = blockIdx.y % num_box; 
	const int box_row_idx = box_idx / box_num_size;
	const int box_col_idx = box_idx % box_num_size;

	///反向求的输入width
	const int in_size = box_in_size * box_num_size; 
	int in_pixs = in_size * in_size;
	int pool_pixs = pool_forward_size * pool_forward_size;

	//输出的行列idx,当pooling之后的width小于32时，box_row_idx会等于0
	int out_row = box_out_size * box_row_idx + threadIdx.y;
	int out_col = box_out_size * box_col_idx + threadIdx.x;

	if(out_row < pool_forward_size && out_col < pool_forward_size){
		//计算本块pooling对应的输入conv块起始位置
		//本线程对应的pooling值
		targets += img_idx * num_filters * in_pixs + filt_idx * in_pixs \
				   + box_row_idx * box_in_size * in_size \
				   + box_col_idx * box_in_size;
		dE_dy_i += img_idx * num_filters * pool_pixs + filt_idx * pool_pixs \
				   + out_row * pool_forward_size + out_col;

		float ele;
		int interval = pool_forward_size - box_out_size * (box_num_size - 1);
		for(int i = threadIdx.x; i < box_in_size; i+=interval)
			for(int j = threadIdx.y; j < box_in_size; j+=interval)
				result[i*box_in_size + j]	= 0;

		for(int i = 0; i < avg_pool_size; i++){
			for(int j = 0; j < avg_pool_size; j++){
				int box_in_row = threadIdx.y * stride + i;
				int box_in_col = threadIdx.x * stride + j;
				if(box_in_row < box_in_size && box_in_col < box_in_size){
					ele = dE_dy_i[0] / (avg_pool_size * avg_pool_size);
					__syncthreads();
					atomicAdd(result + box_in_row*box_in_size+box_in_col, ele);
					__syncthreads();
				}
			}
		}
		for(int i = threadIdx.y; i < box_in_size; i+=interval)
			for(int j = threadIdx.x; j < box_in_size; j+=interval)
				targets[i * in_size + j] = result[i * box_in_size + j];
	}

}

__global__ void compute_dE_db(const float* dE_dy, float* dE_db_h, \
		const int conv_forward_size) {
	extern __shared__ float result[];

	const int idx = threadIdx.x + blockDim.x * threadIdx.y; 
	const int filt_idx = blockIdx.y;
	const int img_idx = blockIdx.x;
	const int num_filters = gridDim.y;

	if(idx == 0)
		result[0] = 0;
	//某一张24*24的起始位置，本函数是将这24*24个点全部加起来最后生成一个点
	const int filt_pixs = conv_forward_size * conv_forward_size;
	dE_dy += img_idx * num_filters * filt_pixs + filt_idx * filt_pixs + threadIdx.x;

	float ele = dE_dy[0];

	__syncthreads();
	atomicAdd(result, ele);
	__syncthreads();

	if (idx == 0) {
		dE_db_h[img_idx * num_filters + filt_idx] = result[0] / filt_pixs;
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
		const int in_size, const int in_channel, const int overlap_len, \
		const int box_in_size, const int box_num_size, const int stride){

	const int img_idx = blockIdx.x;
	const int filt_idx = threadIdx.x;

	const int unfold_in_size = box_in_size * box_num_size; 
	const int in_pixs = in_size * in_size;
	const int unfold_in_pix = unfold_in_size*unfold_in_size;

	src += img_idx*in_channel*unfold_in_pix + filt_idx*unfold_in_pix;
	targets += img_idx*in_channel*in_pixs + filt_idx*in_pixs;

	const int stride_for_overlap = box_in_size - stride;

	for(int i = 0; i < in_size; i++){
		for(int j = 0; j < in_size; j++){
			int unfold_in_row = i + overlap_len*(i/stride_for_overlap);
			int unfold_in_col = j + overlap_len*(j/stride_for_overlap);
			src[i*in_size+j] += targets[unfold_in_row*unfold_in_size+unfold_in_col];

		}
	}
}

















