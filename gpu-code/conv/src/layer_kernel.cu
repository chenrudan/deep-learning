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

__global__ void ori_to_padding(const float* src, float* dst, const int numKernels, \
		const int img_size, const int padded_img_size, const int img_channel){

	int index;
	int imgPixs = img_size * img_size;
	int paddedImgPixs = padded_img_size * padded_img_size;
	int pad = (padded_img_size - img_size) / 2;
	CUDA_KERNEL_LOOP(idx, numKernels){

		const int imgIdx = idx / (img_channel * imgPixs);
		const int srcCol = idx % (img_channel * imgPixs);
		const int imgChannelIdx = srcCol / imgPixs;
		const int imgRow = (srcCol % imgPixs) / img_size; 
		const int imgCol = (srcCol % imgPixs) % img_size; 
		index = imgIdx * img_channel * paddedImgPixs + imgChannelIdx * paddedImgPixs \
					+ (imgRow + pad) * padded_img_size \
					+ (imgCol + pad); 
		dst[index] = src[idx]; 
	}

}



__global__ void im2col_img(const float* conv_result, float* targets, \
		const int numKernels, const int img_size, const int filter_channel, \
		const int img_channel, const int filter_size, const int conv_forward_size, \
		const int conv_stride){

	const int imgPixs = img_size * img_size;
	const int convPixs = conv_forward_size * conv_forward_size;
	const int widthNoChannel = filter_size * filter_size;
	const int width = widthNoChannel * filter_channel;
	int index;
	//这个numkernel比conv要大
	CUDA_KERNEL_LOOP(idx, numKernels){
		//此处的width指的是5*5*16,height指的是100*32*32
		const int filtChannelIdx = (idx % width) / widthNoChannel;
		//widthIdx指的是filt*filt的id
		const int widthIdx = (idx % width) % widthNoChannel;
		const int filtRow = widthIdx / filter_size;
		const int filtCol = widthIdx % filter_size;

		const int imgIdx = (idx / width) / imgPixs;
		const int heightIdx = (idx / width) % imgPixs;
		const int imgRow = heightIdx / img_size;
		const int imgCol = heightIdx % img_size;

		index =	imgIdx * filter_channel * convPixs \
				+ filtChannelIdx * convPixs;
		if((imgRow % conv_stride == filtRow % conv_stride) && \
				(imgCol % conv_stride == filtCol % conv_stride)){
			const int convRow = (imgRow - filtRow) / conv_stride;
			const int convCol = (imgCol - filtCol) / conv_stride; 
			if(convRow >= 0 && convRow < conv_forward_size \
					&& convCol >= 0 && convCol < conv_forward_size){
				index += convRow * conv_forward_size + convCol;
				targets[idx] = conv_result[index];
			}
		}
	}
	__syncthreads();

}

__global__ void im2col_filt(const float* imgs, float* targets, \
		const int numKernels, const int img_size,  \
		const int img_channel, \
		const int filter_size, const int conv_forward_size, \
		const int conv_step_size){

	int index;
	CUDA_KERNEL_LOOP(idx, numKernels){
		int imgPixs = img_size * img_size;
		int filPixs = filter_size * filter_size;
		int width = filPixs * img_channel;
		int heightNoBatch = conv_forward_size * conv_forward_size;

		//此处的width指的是5*5*3,height指的是24*24
		const int channelIdx = (idx % width) / filPixs;
		//widthIdx指的是5*5的id
		const int widthIdx = (idx % width) % filPixs;
		const int filtRow = widthIdx / filter_size;
		const int filtCol = widthIdx % filter_size;
		const int imgIdx = (idx / width) / heightNoBatch;
		const int heightIdx = (idx / width) % heightNoBatch;
		const int convRow = heightIdx / conv_forward_size;
		const int convCol = heightIdx % conv_forward_size;
		//输入图片的位置
		const int imgRow = convRow * conv_step_size + filtRow;
		const int imgCol = convCol * conv_step_size + filtCol;
		index = imgIdx * img_channel * imgPixs + channelIdx * imgPixs \
				+ imgRow * img_size \
				+ imgCol; 
			//输出图片的位置
		targets[idx] = imgs[index];
	}
	__syncthreads();

}

__global__ void im2col_conv(const float* imgs, float* targets, \
		const int numKernels, const int minibatch, const int img_size, \
		const int img_channel, const int filter_size, const int conv_forward_size, \
		const int conv_step_size){

	//行表示为minibatch*convsize*convsize*inchannel
	int index;
	CUDA_KERNEL_LOOP(idx, numKernels){
		int imgPixs = img_size * img_size;
		int widthNoBatch = conv_forward_size * conv_forward_size;
		int width = widthNoBatch * minibatch;
		int heightNoChannel = filter_size * filter_size;

		//此处的width指的是100*24*24,height指的是5*5*3
		const int imgIdx = (idx % width) / widthNoBatch;
		//widthIdx指的是conv*conv的id
		const int widthIdx = (idx % width) % widthNoBatch;
		const int convRow = widthIdx / conv_forward_size;
		const int convCol = widthIdx % conv_forward_size;

		const int channelIdx = (idx / width) / heightNoChannel;
		const int heightIdx = (idx / width) % heightNoChannel;
		const int filtRow = heightIdx / filter_size;
		const int filtCol = heightIdx % filter_size;
		//输入图片的位置
		const int imgRow = convRow * conv_step_size + filtRow;
		const int imgCol = convCol * conv_step_size + filtCol;
		index = imgIdx * img_channel * imgPixs + channelIdx * imgPixs \
				+ imgRow * img_size \
				+ imgCol; 
			//输出图片的位置
		targets[idx] = imgs[index];
	}
	__syncthreads();

}

__global__ void reshape_w(float* un_w, const float* w, \
		const int numKernels, const int filter_size, \
		const int filter_channel, const int img_channel){

	int index;

	CUDA_KERNEL_LOOP(idx, numKernels){

		int filPixs = filter_size * filter_size;
		const int dstRow = idx / img_channel;
		const int dstCol = idx % img_channel;
		const int oriRow = dstCol * filPixs + dstRow % filPixs;
		const int oriCol = dstRow / filPixs;
		index = oriRow * filter_channel + oriCol;
		un_w[idx] = w[index]; 
	}
}

__global__ void reshape_In(float* in, const float* un_in, \
		const int numKernels, const int in_size, \
		const int padded_img_size, const int img_channel){

	int index;
	int imgPixs = in_size * in_size;
	int paddedImgPixs = padded_img_size * padded_img_size;
	int pad = (padded_img_size - in_size) / 2;
	CUDA_KERNEL_LOOP(idx, numKernels){

		const int dstRow = idx / img_channel;
		const int dstCol = idx % img_channel;
		const int imgIdx = dstRow / paddedImgPixs;
		const int imgRow = (dstRow % paddedImgPixs) / padded_img_size; 
		const int imgCol = (dstRow % paddedImgPixs) % padded_img_size; 
		const int lowerBound = pad;
		const int higherBound = padded_img_size - pad;
		if(imgRow >= lowerBound && imgRow < higherBound \
			&& imgCol >= lowerBound && imgCol < higherBound){
				index = imgIdx * img_channel * imgPixs + dstCol * imgPixs \
						+ (imgRow - pad) * in_size \
						+ (imgCol - pad); 
			in[index] = un_in[idx]; 
		}
	}
}


__global__ void reshape_y(const float* un_y_h, float* y_h, \
		const int numKernels, const int conv_forward_size, \
		const int filter_channel){

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < numKernels){
		int convPixs = conv_forward_size * conv_forward_size;

		const int dstRow = idx / (convPixs * filter_channel);
		const int dstCol = idx % (convPixs * filter_channel);
		const int oriCol = dstCol / convPixs;
		const int oriRow = dstRow * convPixs + dstCol % convPixs;
		un_y_h += oriRow * filter_channel + oriCol;
		y_h[idx] = sigmoid(un_y_h[0]); 
		//		y_h[idx] = (un_y_h[0]); 
	}

}

__global__ void reshape_dE_dx_sigmoid(float* un_dE_dx_h, const float* dE_dx_h, \
		const int numKernels, const int conv_forward_size, \
		const int filter_channel){
	int index;
	CUDA_KERNEL_LOOP(idx, numKernels){
		int convPixs = conv_forward_size * conv_forward_size;

		const int dstRow = idx / (convPixs * filter_channel);
		const int dstCol = idx % (convPixs * filter_channel);
		const int oriCol = dstCol / convPixs;
		const int oriRow = dstRow * convPixs + dstCol % convPixs;
		index = oriRow * filter_channel + oriCol;
		un_dE_dx_h[index] = dE_dx_h[idx]; 
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



__global__ void max_pooling(float* convOutputs, float* targets, int* maxPoolPos, \
		const int conv_forward_size, const int pool_forward_size, \
		const int max_pool_size, const int stride){
	const int numFilters = gridDim.y;
	const int imgIdx = blockIdx.x;
	const int filtIdx = blockIdx.y;

	extern __shared__ float shFeatureMap[];

	int convPixs = conv_forward_size * conv_forward_size;
	int poolPixs = pool_forward_size * pool_forward_size;

	if(threadIdx.x < pool_forward_size && threadIdx.y < pool_forward_size){
		convOutputs += imgIdx * numFilters * convPixs + filtIdx * convPixs; 
		targets += imgIdx * numFilters * poolPixs + filtIdx * poolPixs \
				   + threadIdx.y * pool_forward_size + threadIdx.x;
		maxPoolPos += imgIdx * numFilters * poolPixs + filtIdx * poolPixs \
					  + threadIdx.y * pool_forward_size + threadIdx.x;

		for(int i = 0; i < max_pool_size; i++){
			for(int j = 0; j < max_pool_size; j++){
				int convRow = threadIdx.y * stride + i;
				int convCol = threadIdx.x * stride + j;
				int shRow = threadIdx.y * max_pool_size + i;
				int shCol = threadIdx.x * max_pool_size + j;
				if(convRow < conv_forward_size && convCol < conv_forward_size){
					shFeatureMap[shRow * pool_forward_size * max_pool_size + shCol] \
						= convOutputs[convRow * conv_forward_size + convCol];
				}
				else{
					shFeatureMap[shRow * pool_forward_size * max_pool_size + shCol] \
						= 0;
				}
			}
		}
		__syncthreads();

		float *myShFM = &shFeatureMap[0];
		myShFM +=  threadIdx.y * max_pool_size * pool_forward_size * max_pool_size \
				   + threadIdx.x * max_pool_size;

		float max_value = myShFM[0];
		int max_pos = 0;
		for(int i = 0; i < max_pool_size; i++){
			for(int j = 0; j < max_pool_size; j++){
				if(myShFM[i * max_pool_size * pool_forward_size + j] > max_value){
					max_value = myShFM[i * max_pool_size * pool_forward_size + j];
					max_pos = i * max_pool_size + j;
				}
			}
		}
		targets[0] = max_value;
		maxPoolPos[0] = max_pos;
	}
}


__global__ void compute_dE_dy_max(float* dE_dy_i, float* out, int* maxPoolPos, \
		const int conv_forward_size, const int pool_forward_size, \
		const int max_pool_size, const int stride){

	extern __shared__ float result[];

	int convPixs = conv_forward_size * conv_forward_size;
	int poolPixs = pool_forward_size * pool_forward_size;

	const int numFilters = gridDim.y;
	const int imgIdx = blockIdx.x;
	const int filtIdx = blockIdx.y;

	int posIdx= threadIdx.y * conv_forward_size * stride \
	               + threadIdx.x * stride;

	if(threadIdx.x < pool_forward_size && threadIdx.y < pool_forward_size){
		out += imgIdx * numFilters * convPixs + filtIdx * convPixs \
			   + threadIdx.y * conv_forward_size * stride \
			   + threadIdx.x * stride; 
		dE_dy_i += imgIdx * numFilters * poolPixs + filtIdx * poolPixs \
				   + threadIdx.y * pool_forward_size + threadIdx.x;
		maxPoolPos += imgIdx * numFilters * poolPixs + filtIdx * poolPixs \
					  + threadIdx.y * pool_forward_size + threadIdx.x;
		
		for(int i = threadIdx.x; i < conv_forward_size; i+=pool_forward_size){
			for(int j = threadIdx.y; j < conv_forward_size; j+=pool_forward_size){
				result[i*conv_forward_size + j]	= 0;
			}
		}

		int pos = maxPoolPos[0];
		int row = pos / max_pool_size;
		int col = pos % max_pool_size;

		posIdx += row * conv_forward_size + col;

		float ele = dE_dy_i[0];

		__syncthreads();
		atomicAdd(result+posIdx, ele);
		__syncthreads();

		if(row <= conv_forward_size && col <= conv_forward_size){
			out[row * conv_forward_size + col] = result[posIdx];
		}
	}
}

__global__ void compute_dE_db(const float* dE_dx_h, float* dE_db_h, \
		const int conv_forward_size) {
	extern __shared__ float result[];

	const int idx = threadIdx.x + blockDim.x * threadIdx.y; 
	const int filtIdx = blockIdx.y;
	const int imgIdx = blockIdx.x;
	const int numFilters = gridDim.y;

	if(idx == 0)
		result[0] = 0;
	//某一张24*24的起始位置，本函数是将这24*24个点全部加起来最后生成一个点
	const int filPixs = conv_forward_size * conv_forward_size;
	dE_dx_h += imgIdx * numFilters * filPixs + filtIdx * filPixs + threadIdx.x;

	float ele = dE_dx_h[0];

	__syncthreads();
	atomicAdd(result, ele);
	__syncthreads();

	if (idx == 0) {
		dE_db_h[imgIdx * numFilters + filtIdx] = result[0] / filPixs;
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















