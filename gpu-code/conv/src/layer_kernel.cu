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

__global__ void im2col_img(const float* conv_result, float* targets, \
		const int num_kernel, const int img_size, const int filter_channel, \
		const int img_channel, const int filter_size, const int conv_forward_size, \
		const int conv_stride){

	const int img_pixs = img_size * img_size;
	const int conv_pixs = conv_forward_size * conv_forward_size;
	const int width_no_channel = filter_size * filter_size;
	const int width = width_no_channel * filter_channel;
	int index;
	//这个numkernel比conv要大
	CUDA_KERNEL_LOOP(idx, num_kernel){
		//此处的width指的是5*5*16,height指的是100*32*32
		const int filt_channel_idx = (idx % width) / width_no_channel;
		//width_idx指的是filt*filt的id
		const int width_idx = (idx % width) % width_no_channel;
		const int filt_row = width_idx / filter_size;
		const int filt_col = width_idx % filter_size;

		const int img_idx = (idx / width) / img_pixs;
		const int heightIdx = (idx / width) % img_pixs;
		const int img_row = heightIdx / img_size;
		const int img_col = heightIdx % img_size;

		index =	img_idx * filter_channel * conv_pixs \
				+ filt_channel_idx * conv_pixs;
		if((img_row % conv_stride == filt_row % conv_stride) && \
				(img_col % conv_stride == filt_col % conv_stride)){
			const int conv_row = (img_row - filt_row) / conv_stride;
			const int conv_col = (img_col - filt_col) / conv_stride; 
			if(conv_row >= 0 && conv_row < conv_forward_size \
					&& conv_col >= 0 && conv_col < conv_forward_size){
				index += conv_row * conv_forward_size + conv_col;
				targets[idx] = conv_result[index];
			}
		}
	}
	__syncthreads();

}

__global__ void im2col_filt(const float* imgs, float* targets, \
		const int num_kernel, const int img_size,  \
		const int img_channel, \
		const int filter_size, const int conv_forward_size, \
		const int conv_step_size){

	int index;
	CUDA_KERNEL_LOOP(idx, num_kernel){
		int img_pixs = img_size * img_size;
		int filt_pixs = filter_size * filter_size;
		int width = filt_pixs * img_channel;
		int heightNoBatch = conv_forward_size * conv_forward_size;

		//此处的width指的是5*5*3,height指的是24*24
		const int channelIdx = (idx % width) / filt_pixs;
		//width_idx指的是5*5的id
		const int width_idx = (idx % width) % filt_pixs;
		const int filt_row = width_idx / filter_size;
		const int filt_col = width_idx % filter_size;
		const int img_idx = (idx / width) / heightNoBatch;
		const int heightIdx = (idx / width) % heightNoBatch;
		const int conv_row = heightIdx / conv_forward_size;
		const int conv_col = heightIdx % conv_forward_size;
		//输入图片的位置
		const int img_row = conv_row * conv_step_size + filt_row;
		const int img_col = conv_col * conv_step_size + filt_col;
		index = img_idx * img_channel * img_pixs + channelIdx * img_pixs \
				+ img_row * img_size \
				+ img_col; 
		//输出图片的位置
		targets[idx] = imgs[index];
	}
	__syncthreads();

}

__global__ void im2col_conv(const float* imgs, float* targets, \
		const int num_kernel, const int minibatch, const int img_size, \
		const int img_channel, const int filter_size, const int conv_forward_size, \
		const int conv_step_size){

	//行表示为minibatch*convsize*convsize*inchannel
	int index;
	CUDA_KERNEL_LOOP(idx, num_kernel){
		int img_pixs = img_size * img_size;
		int widthNoBatch = conv_forward_size * conv_forward_size;
		int width = widthNoBatch * minibatch;
		int heightNoChannel = filter_size * filter_size;

		//此处的width指的是100*24*24,height指的是5*5*3
		const int img_idx = (idx % width) / widthNoBatch;
		//width_idx指的是conv*conv的id
		const int width_idx = (idx % width) % widthNoBatch;
		const int conv_row = width_idx / conv_forward_size;
		const int conv_col = width_idx % conv_forward_size;

		const int channelIdx = (idx / width) / heightNoChannel;
		const int heightIdx = (idx / width) % heightNoChannel;
		const int filt_row = heightIdx / filter_size;
		const int filt_col = heightIdx % filter_size;
		//输入图片的位置
		const int img_row = conv_row * conv_step_size + filt_row;
		const int img_col = conv_col * conv_step_size + filt_col;
		index = img_idx * img_channel * img_pixs + channelIdx * img_pixs \
				+ img_row * img_size \
				+ img_col; 
		//输出图片的位置
		targets[idx] = imgs[index];
	}
	__syncthreads();

}

__global__ void reshape_w(float* un_w, const float* w, \
		const int num_kernel, const int filter_size, \
		const int filter_channel, const int img_channel){

	int index;

	CUDA_KERNEL_LOOP(idx, num_kernel){

		int filt_pixs = filter_size * filter_size;
		const int dst_row = idx / img_channel;
		const int dst_col = idx % img_channel;
		const int ori_row = dst_col * filt_pixs + dst_row % filt_pixs;
		const int ori_col = dst_row / filt_pixs;
		index = ori_row * filter_channel + ori_col;
		un_w[idx] = w[index]; 
	}
}

__global__ void reshape_In(float* in, const float* un_in, \
		const int num_kernel, const int in_size, \
		const int padded_img_size, const int img_channel){

	int index;
	int img_pixs = in_size * in_size;
	int padded_img_pixs = padded_img_size * padded_img_size;
	int pad = (padded_img_size - in_size) / 2;
	CUDA_KERNEL_LOOP(idx, num_kernel){

		const int dst_row = idx / img_channel;
		const int dst_col = idx % img_channel;
		const int img_idx = dst_row / padded_img_pixs;
		const int img_row = (dst_row % padded_img_pixs) / padded_img_size; 
		const int img_col = (dst_row % padded_img_pixs) % padded_img_size; 
		const int lowerBound = pad;
		const int higherBound = padded_img_size - pad;
		if(img_row >= lowerBound && img_row < higherBound \
				&& img_col >= lowerBound && img_col < higherBound){
			index = img_idx * img_channel * img_pixs + dst_col * img_pixs \
					+ (img_row - pad) * in_size \
					+ (img_col - pad); 
			in[index] = un_in[idx]; 
		}
	}
}


__global__ void reshape_y(const float* un_y, float* y, \
		const int num_kernel, const int conv_forward_size, \
		const int filter_channel){

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < num_kernel){
		int conv_pixs = conv_forward_size * conv_forward_size;

		const int dst_row = idx / (conv_pixs * filter_channel);
		const int dst_col = idx % (conv_pixs * filter_channel);
		const int ori_col = dst_col / conv_pixs;
		const int ori_row = dst_row * conv_pixs + dst_col % conv_pixs;
		un_y += ori_row * filter_channel + ori_col;
		//y[idx] = sigmoid(un_y[0]); 
		y[idx] = (un_y[0]); 
	}

}

__global__ void reshape_dE_dy(float* un_dE_dy, const float* dE_dy, \
		const int num_kernel, const int conv_forward_size, \
		const int filter_channel){
	int index;
	CUDA_KERNEL_LOOP(idx, num_kernel){
		int conv_pixs = conv_forward_size * conv_forward_size;

		const int dst_row = idx / (conv_pixs * filter_channel);
		const int dst_col = idx % (conv_pixs * filter_channel);
		const int ori_col = dst_col / conv_pixs;
		const int ori_row = dst_row * conv_pixs + dst_col % conv_pixs;
		index = ori_row * filter_channel + ori_col;
		un_dE_dy[index] = dE_dy[idx]; 
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

//row-major
__global__ void compute_dE_dy_j(const float* y_j, const float* labels, \
		float* dE_dy_j, const int width) {
	const int tx = blockIdx.x;
	const int ty = blockIdx.x * width + threadIdx.x;

	const int lab = labels[tx];

	if(threadIdx.x < width)
		dE_dy_j[ty] = y_j[ty] - (lab == threadIdx.x);
	__syncthreads();
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

	extern __shared__ float shFeatureMap[];

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

		for(int i = 0; i < max_pool_size; i++){
			for(int j = 0; j < max_pool_size; j++){
				int conv_row = out_row * stride + i;
				int conv_col = out_col * stride + j;
				int shRow = threadIdx.y * max_pool_size + i;
				int shCol = threadIdx.x * max_pool_size + j;
				if(conv_row < conv_forward_size && conv_col < conv_forward_size){
					shFeatureMap[shRow * box_out_size * max_pool_size + shCol] \
						= convOutputs[conv_row * conv_forward_size + conv_col];
				}
				else{
					shFeatureMap[shRow * box_out_size * max_pool_size + shCol] \
						= 0;
				}
			}
		}
		__syncthreads();

		float *myShFM = &shFeatureMap[0];
		myShFM +=  threadIdx.y * max_pool_size * box_out_size * max_pool_size \
				   + threadIdx.x * max_pool_size;

		float max_value = myShFM[0];
		int max_pos = 0;
		for(int i = 0; i < max_pool_size; i++){
			for(int j = 0; j < max_pool_size; j++){
				if(myShFM[i * max_pool_size * box_out_size + j] > max_value){
					max_value = myShFM[i * max_pool_size * box_out_size + j];
					max_pos = i * max_pool_size + j;
				}
			}
		}
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

	extern __shared__ float shFeatureMap[];

	int conv_pixs = conv_forward_size * conv_forward_size;
	int pool_pixs = pool_forward_size * pool_forward_size;

	int out_row = MAX_THREAD_SIZE * box_row_idx + threadIdx.y;
	int out_col = MAX_THREAD_SIZE * box_col_idx + threadIdx.x;

	if(out_row < pool_forward_size && out_col < pool_forward_size){
		convOutputs += img_idx * in_channels * conv_pixs + filt_idx * conv_pixs; 
		targets += img_idx * in_channels * pool_pixs + filt_idx * pool_pixs \
				   + out_row * pool_forward_size + out_col;

		for(int i = 0; i < avg_pool_size; i++){
			for(int j = 0; j < avg_pool_size; j++){
				int conv_row = out_row * stride + i;
				int conv_col = out_col * stride + j;
				int shRow = threadIdx.y * avg_pool_size + i;
				int shCol = threadIdx.x * avg_pool_size + j;
				if(conv_row < conv_forward_size && conv_col < conv_forward_size){
					shFeatureMap[shRow * box_out_size * avg_pool_size + shCol] \
						= convOutputs[conv_row * conv_forward_size + conv_col];
				}
				else{
					shFeatureMap[shRow * box_out_size * avg_pool_size + shCol] \
						= 0;
				}
			}
		}
		__syncthreads();

		float *myShFM = &shFeatureMap[0];
		//计算当前一个输出点的输入位置
		myShFM +=  threadIdx.y * avg_pool_size * box_out_size * avg_pool_size \
				   + threadIdx.x * avg_pool_size;

		float avg_value = 0;
		for(int i = 0; i < avg_pool_size; i++){
			for(int j = 0; j < avg_pool_size; j++){
				avg_value += myShFM[i * avg_pool_size * box_out_size + j];
			}
		}
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
	int out_row = MAX_THREAD_SIZE * box_row_idx + threadIdx.y;
	int out_col = MAX_THREAD_SIZE * box_col_idx + threadIdx.x;

	int in_row = box_in_size * box_row_idx + threadIdx.y * stride;
   	int	in_col = box_in_size * box_col_idx + threadIdx.x * stride;

	//共享内存的大小只有计算一块pool输出的对应conv输入
	extern __shared__ float result[];

	int posIdx= threadIdx.y * box_in_size * stride \
				+ threadIdx.x * stride;

	if(out_row < pool_forward_size && out_col < pool_forward_size){
		targets += img_idx * num_filters * in_pixs + filt_idx * in_pixs \
			   + in_row * in_size + in_col; 
		dE_dy_i += img_idx * num_filters * pool_pixs + filt_idx * pool_pixs \
				   + out_row * pool_forward_size + out_col;
		maxPoolPos += img_idx * num_filters * pool_pixs + filt_idx * pool_pixs \
					  + out_row * pool_forward_size + out_col;

		for(int i = threadIdx.x; i < box_in_size; i+=box_out_size)
			for(int j = threadIdx.y; j < box_in_size; j+=box_out_size)
				result[i*box_in_size + j] = 0;

		int pos = maxPoolPos[0];
		int row = pos / max_pool_size;
		int col = pos % max_pool_size;

		posIdx += row * box_in_size + col;

		float ele = dE_dy_i[0];

		__syncthreads();
		atomicAdd(result+posIdx, ele);
		__syncthreads();

		targets[row * in_size + col] = result[posIdx];
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
	int out_row = MAX_THREAD_SIZE * box_row_idx + threadIdx.y;
	int out_col = MAX_THREAD_SIZE * box_col_idx + threadIdx.x;


	if(out_row < pool_forward_size && out_col < pool_forward_size){
		//计算本块pooling对应的输入conv块起始位置
		//本线程对应的pooling值
		targets += img_idx * num_filters * in_pixs + filt_idx * in_pixs \
				   	+ box_row_idx * box_in_size * in_size \
			   	 	+ box_col_idx * box_in_size;
		dE_dy_i += img_idx * num_filters * pool_pixs + filt_idx * pool_pixs \
				   + out_row * pool_forward_size + out_col;

		float ele;
		for(int i = threadIdx.x; i < box_in_size; i+=box_out_size)
			for(int j = threadIdx.y; j < box_in_size; j+=box_out_size)
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
		for(int i = threadIdx.y; i < box_in_size; i+=box_out_size)
			for(int j = threadIdx.x; j < box_in_size; j+=box_out_size)
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
__global__ void compute_dE_dy(const float* y_j, const float* labels, \
		float* dE_dy_j, const int width) {
	const int tx = blockIdx.x;
	const int ty = blockIdx.x * width + threadIdx.x;

	const int lab = labels[tx];

	if(threadIdx.x < width)
		dE_dy_j[ty] = y_j[ty] - (lab == threadIdx.x);
	__syncthreads();
}


__global__ void compactOverlap(float* src, float* targets, \
		const int in_size, const int com_stride, const int com_len, \
		const int out_size, const int out_channel){

	const int filt_idx = blockIdx.y;
	const int img_idx = blockIdx.x;

	const int in_pixs = in_size * in_size;
	const int out_pixs = out_size * out_size;

	int out_row = threadIdx.y;
	int out_col = threadIdx.x;
	int in_row = out_row - com_len * (out_row / com_stride);
	int in_col = out_col - com_len * (out_col / com_stride);

	src += img_idx * out_channel * out_pixs + filt_idx * out_pixs;
	targets += img_idx * out_channel * in_pixs + filt_idx * in_pixs;

	extern __shared__ float result[];

	for(int i = out_row; in_row < in_size; i += blockDim.y, in_row = i - com_len * (i / com_stride)){
		for(int j = out_col; in_row < in_col; j += blockDim.x, in_col = j - com_len * (j / com_stride)){
			
			__syncthreads();
			atomicAdd(result+in_row * in_size + in_col, src[i*out_size + j]);
			__syncthreads();
		}
	}

	for(int i = threadIdx.y; i < in_size; i += blockDim.y){
		for(int j = threadIdx.x; j < in_size; j += blockDim.x){
			targets[i * in_size + j] = result[i * in_size + j];
		}
	}
}

















