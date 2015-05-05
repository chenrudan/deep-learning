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



__global__ void im2col_img(const float* conv_result, float* targets, \
		const int numKernels, const int widthNoChannel, const int width, \
		const int img_size, const int filter_channel, \
		const int img_channel, const int filter_size, const int conv_forward_size, \
		const int conv_stride){

	const int imgPixs = img_size * img_size;
	const int convPixs = conv_forward_size * conv_forward_size;
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
			//			else
			//				targets[idx] = 0;
		}
		//		else
		//			targets[idx] = 0;
		//输出图片的位置
	}
	__syncthreads();

}

__global__ void im2col_filt(const float* imgs, float* targets, \
		const int numKernels, const int widthNoChannel, const int width, \
		const int heightNoBatch, const int img_size, const int img_channel, \
		const int filter_size, const int conv_forward_size, \
		const int conv_step_size){

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < numKernels){
		int imgPixs = img_size * img_size;
		int filPixs = filter_size * filter_size;

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
		imgs += imgIdx * img_channel * imgPixs + channelIdx * imgPixs \
				+ (convRow * conv_step_size + filtRow) * img_size \
				+ (convCol * conv_step_size + filtCol); 
		//输出图片的位置
		targets[idx] = imgs[0];
	}
	__syncthreads();

}

__global__ void im2col_conv(const float* imgs, float* targets, \
		const int numKernels, const int widthNoBatch, const int width, \
		const int heightNoChannel, const int img_size, \
		const int img_channel, const int filter_size, const int conv_forward_size, \
		const int conv_step_size){

	//行表示为minibatch*convsize*convsize*inchannel
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < numKernels){
		int imgPixs = img_size * img_size;
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
		imgs += imgIdx * img_channel * imgPixs + channelIdx * imgPixs \
				+ (convRow * conv_step_size + filtRow) * img_size \
				+ (convCol * conv_step_size + filtCol); 
		//输出图片的位置
		targets[idx] = imgs[0];
	}
	__syncthreads();

}

__global__ void reshape_w(float* un_w, const float* w, \
		const int numKernels, const int filter_size, \
		const int filter_channel, const int img_channel){

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < numKernels){

		int filPixs = filter_size * filter_size;
		const int dstRow = idx / img_channel;
		const int dstCol = idx % img_channel;
		const int oriRow = dstCol * filPixs + dstRow % filPixs;
		const int oriCol = dstRow / filPixs;
		w += oriRow * filter_channel + oriCol;
		un_w[idx] = w[0]; 
	}
}

__global__ void reshape_In(float* in, const float* un_in, \
		const int numKernels, const int in_size, \
		const int img_channel){

	int index;
	CUDA_KERNEL_LOOP(idx, numKernels){

		int imgPixs = in_size * in_size;
		const int dstRow = idx / img_channel;
		const int dstCol = idx % img_channel;
		const int oriCol = dstCol * imgPixs + dstRow % imgPixs;
		const int oriRow = dstRow / imgPixs;
		index = oriRow * imgPixs * img_channel + oriCol;
		in[index] = un_in[idx]; 
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
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < numKernels){
		int convPixs = conv_forward_size * conv_forward_size;

		const int dstRow = idx / (convPixs * filter_channel);
		const int dstCol = idx % (convPixs * filter_channel);
		const int oriCol = dstCol / convPixs;
		const int oriRow = dstRow * convPixs + dstCol % convPixs;
		un_dE_dx_h += oriRow * filter_channel + oriCol;
		un_dE_dx_h[0] = dE_dx_h[idx]; 
	}
}

/*
   __global__ void convolution_forward(const float* imgs, const float* filters, \
   const float* biases, float* targets, const int filConvtimes, \
   const int imgConvtimes) {
   int imgPixs = IMG_SIZE * IMG_SIZE;
   int filPixs = FILTER_SIZE * FILTER_SIZE;
   int convPixs = CONV_FORWARD_SIZE * CONV_FORWARD_SIZE;
//一个block计算一张图与一个featuremap的卷积，开辟了28个线程，每个线程算28个值
//放在shared memory里面的数据，featuremap的参数是要共享的

//	__shared__ struct conv tmp;
//	__device__ float* value = new float[filPixs];
//	tmp.data = value;
__shared__ float shImg[IMG_SIZE][IMG_SIZE];
__shared__ float shFilter[FILTER_SIZE][FILTER_SIZE];
__shared__ float shBias;

const int imgIdx = blockIdx.x;
const int filtIdx = blockIdx.y;
const int numFilters = gridDim.y;

//只是给shBias找个机会赋值而已
if(threadIdx.x + threadIdx.y == 0) {
shBias = biases[filtIdx];
}

//为了得到需要计算的image和filter和target的数据起始点
imgs += imgIdx * imgPixs;
filters += filtIdx * filPixs;
targets += imgIdx * numFilters * convPixs + filtIdx * convPixs \
+ threadIdx.y * CONV_FORWARD_SIZE + threadIdx.x;

//多线程复制数据到sm里面
for(int i = 0; i < imgConvtimes + 1; i++){
for(int j = 0; j < imgConvtimes + 1; j++){
int col = threadIdx.x + blockDim.x * i;
int row = threadIdx.y + blockDim.y * j;	
if((row < IMG_SIZE) && (col < IMG_SIZE)){
shImg[row][col] = imgs[row * IMG_SIZE + col];
}
}
}
for(int i = 0; i < filConvtimes + 1; i++){
for(int j = 0; j < filConvtimes + 1; j++){
int col = threadIdx.x + blockDim.x * i;
int row = threadIdx.y + blockDim.y * j;	
if((row < FILTER_SIZE) && (col < FILTER_SIZE)){
shFilter[row][col] = filters[row * FILTER_SIZE + col];
}
}
}

__syncthreads();

float *myShImg = &shImg[0][0];
myShImg += threadIdx.y * IMG_SIZE + threadIdx.x;
float prod = shBias;

for(int i = 0; i < FILTER_SIZE; i++){
for(int j = 0; j < FILTER_SIZE; j++){
prod += shFilter[i][j] * myShImg[i * IMG_SIZE + j];
}
}
__syncthreads();

targets[0] = sigmoid(prod);
//targets[0] = prod;
}

__global__ void avg_pooling(float* convOutputs, float* targets){
const int numFilters = gridDim.y;
const int imgIdx = blockIdx.x;
const int filtIdx = blockIdx.y;

__shared__ float shFeatureMap[CONV_FORWARD_SIZE][CONV_FORWARD_SIZE];

int convPixs = CONV_FORWARD_SIZE * CONV_FORWARD_SIZE;
int poolPixs = POOL_FORWARD_SIZE * POOL_FORWARD_SIZE;
convOutputs += imgIdx * numFilters * convPixs + filtIdx * convPixs; 
targets += imgIdx * numFilters * poolPixs + filtIdx * poolPixs \
		   + threadIdx.y * POOL_FORWARD_SIZE + threadIdx.x;

if((blockDim.x > CONV_FORWARD_SIZE) && (blockDim.y > CONV_FORWARD_SIZE) \
		&& (threadIdx.x < CONV_FORWARD_SIZE) && (threadIdx.y < CONV_FORWARD_SIZE)){
	shFeatureMap[threadIdx.y][threadIdx.x] = \
											 convOutputs[threadIdx.y * CONV_FORWARD_SIZE + threadIdx.x];
}
if((blockDim.x <= CONV_FORWARD_SIZE) && (blockDim.y <= CONV_FORWARD_SIZE)){
	int dist = CONV_FORWARD_SIZE - blockDim.x;

	shFeatureMap[threadIdx.y][threadIdx.x] = \
											 convOutputs[threadIdx.y * CONV_FORWARD_SIZE + threadIdx.x];
	if(threadIdx.y < dist){
		shFeatureMap[threadIdx.y + blockDim.x][threadIdx.x] = \
															  convOutputs[(threadIdx.y + blockDim.x) * CONV_FORWARD_SIZE \
															  + threadIdx.x];
	}
	if(threadIdx.x < dist){
		shFeatureMap[threadIdx.y][threadIdx.x + blockDim.x] = \
															  convOutputs[(threadIdx.y) * CONV_FORWARD_SIZE \
															  + threadIdx.x + blockDim.x];
	}
	if(threadIdx.y < dist && threadIdx.x < dist){
		shFeatureMap[threadIdx.y + blockDim.x][threadIdx.x + blockDim.x] = \
																		   convOutputs[(threadIdx.y + blockDim.x) * CONV_FORWARD_SIZE \
																		   + threadIdx.x + blockDim.x];
	}
}
__syncthreads();

float *myShFM = &shFeatureMap[0][0];
myShFM +=  threadIdx.y * CONV_FORWARD_SIZE * AVG_POOL_Y \
		   + threadIdx.x * AVG_POOL_X;

float avg_value = 0;
for(int i = 0; i < AVG_POOL_X; i++){
	for(int j = 0; j < AVG_POOL_Y; j++){
		avg_value += myShFM[i * CONV_FORWARD_SIZE + j];
	}
}
__syncthreads();
targets[0] = avg_value / (AVG_POOL_X * AVG_POOL_Y);
}
*/
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


/*
   __global__ void compute_dE_dy_h_avg(const float* dE_dy_i, float* out){

   int convPixs = CONV_FORWARD_SIZE * CONV_FORWARD_SIZE;
   int poolPixs = POOL_FORWARD_SIZE * POOL_FORWARD_SIZE;

   const int numFilters = gridDim.y;
   const int imgIdx = blockIdx.x;
   const int filtIdx = blockIdx.y;

   if(threadIdx.x < POOL_FORWARD_SIZE && threadIdx.y < POOL_FORWARD_SIZE){
   out += imgIdx * numFilters * convPixs + filtIdx * convPixs \
   + threadIdx.y * CONV_FORWARD_SIZE * AVG_POOL_Y \
   + threadIdx.x * AVG_POOL_X; 
   dE_dy_i += imgIdx * numFilters * poolPixs + filtIdx * poolPixs \
   + threadIdx.y * POOL_FORWARD_SIZE + threadIdx.x;

   for(int i = 0; i < AVG_POOL_X; i++){
   for(int j = 0; j < AVG_POOL_Y; j++){
   out[i * CONV_FORWARD_SIZE + j] 
   = dE_dy_i[0] / (AVG_POOL_X * AVG_POOL_Y); 
   }
   }
   __syncthreads();
   }
   }
   __global__ void convolution_backward(const float* imgs, const float* filters, \
   float* targets, int convFiltimes, int imgFiltimes) {
//filConvtimes指的是filter的大小是卷积结果的多少倍，也就是说是线程总数的多少倍
//通过这么多次的线程重复赋值到shared memory

int imgPixs = IMG_SIZE * IMG_SIZE;
int convPixs = CONV_FORWARD_SIZE * CONV_FORWARD_SIZE;
int filPixs = FILTER_SIZE * FILTER_SIZE;
//一个block计算一张图与一个featuremap的卷积，开辟了28个线程，每个线程算28个值
//放在shared memory里面的数据只有输入图片
//前向卷积生成的输出不共享，直接一张图求一个点

__shared__ float shImg[IMG_SIZE][IMG_SIZE];
__shared__ float shConv[CONV_FORWARD_SIZE][CONV_FORWARD_SIZE];

const int imgIdx = blockIdx.x;
const int filtIdx = blockIdx.y;
const int numFilters = gridDim.y;

//为了得到需要计算的image和filter和target的数据起始点
imgs += imgIdx * imgPixs;
filters += imgIdx * numFilters * convPixs + filtIdx * convPixs;
targets += imgIdx * numFilters * filPixs + filtIdx * filPixs \
+threadIdx.y * FILTER_SIZE + threadIdx.x;
//			   + (FILTER_SIZE - 1 - threadIdx.y) * FILTER_SIZE \
//			   + FILTER_SIZE - 1 - threadIdx.x;

//多线程复制数据到sm里面
for(int i = 0; i < imgFiltimes + 1; i++){
for(int j = 0; j < imgFiltimes + 1; j++){
int col = threadIdx.x + blockDim.x * i;
int row = threadIdx.y + blockDim.y * j;	
if((row < IMG_SIZE) && (col < IMG_SIZE)){
shImg[row][col] = imgs[row * IMG_SIZE + col];
}
}
}
//filp 180
for(int i = 0; i < convFiltimes + 1; i++){
for(int j = 0; j < convFiltimes + 1; j++){
int col = threadIdx.x + blockDim.x * i;
int row = threadIdx.y + blockDim.y * j;	
if((row < CONV_FORWARD_SIZE) && (col < CONV_FORWARD_SIZE)){
//				shConv[CONV_FORWARD_SIZE - 1 - row][CONV_FORWARD_SIZE -1 - col] \
= filters[row * CONV_FORWARD_SIZE + col];
shConv[row][col] \
		= filters[row * CONV_FORWARD_SIZE + col];
}
}
}

__syncthreads();

float *myShImg = &shImg[0][0];
myShImg += threadIdx.y * IMG_SIZE + threadIdx.x;
float prod = 0;

for(int i = 0; i < CONV_FORWARD_SIZE; i++){
	for(int j = 0; j < CONV_FORWARD_SIZE; j++){
		prod += shConv[i][j] * myShImg[i * IMG_SIZE + j];
	}
}
__syncthreads();

targets[0] = prod;

}
*/

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
				int convRow = threadIdx.x * stride + i;
				int convCol = threadIdx.y * stride + j;
				int shRow = threadIdx.x * max_pool_size + i;
				int shCol = threadIdx.y * max_pool_size + j;
				shFeatureMap[shRow * pool_forward_size * max_pool_size + shCol] \
					= convOutputs[convRow * conv_forward_size + convCol];
			}
		}
		__syncthreads();

		float *myShFM = &shFeatureMap[0];
		myShFM +=  threadIdx.y * max_pool_size * pool_forward_size * max_pool_size \
				   + threadIdx.x * max_pool_size;

		float max_value = -10000;
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

		out[row * conv_forward_size + col] = result[posIdx];
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















