/*
 * filename: convnet_kernel.cu
 */

#include <cuda_runtime.h>
#include "convnet_kernel.cuh"

//struct conv{
//	float* data;	
//};

__device__ float logistic(float x) {
	if(x < -300)
		return 0;
	else if( x > 300)
		return 1;
	else
		return 1 / (1 + __expf(-x));
}

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

	targets[0] = logistic(prod);
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


__global__ void max_pooling(float* convOutputs, float* targets, int* maxPoolPos){
	const int numFilters = gridDim.y;
	const int imgIdx = blockIdx.x;
	const int filtIdx = blockIdx.y;

	__shared__ float shFeatureMap[CONV_FORWARD_SIZE][CONV_FORWARD_SIZE];

	int convPixs = CONV_FORWARD_SIZE * CONV_FORWARD_SIZE;
	int poolPixs = POOL_FORWARD_SIZE * POOL_FORWARD_SIZE;
	convOutputs += imgIdx * numFilters * convPixs + filtIdx * convPixs; 
	targets += imgIdx * numFilters * poolPixs + filtIdx * poolPixs \
				   + threadIdx.y * POOL_FORWARD_SIZE + threadIdx.x;
	maxPoolPos += imgIdx * numFilters * poolPixs + filtIdx * poolPixs \
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

	float max_value = -1000;
	int max_pos = 0;
	for(int i = 0; i < AVG_POOL_X; i++){
		for(int j = 0; j < AVG_POOL_Y; j++){
			if(myShFM[i * CONV_FORWARD_SIZE + j] > max_value){
				max_value = myShFM[i * CONV_FORWARD_SIZE + j];
				max_pos = i * AVG_POOL_Y + j;
			}
		}
	}
	targets[0] = max_value;
	maxPoolPos[0] = max_pos;
}


__global__ void compute_dE_dy_h_max(float* dE_dy_i, float* out, int* maxPoolPos){
	
	int convPixs = CONV_FORWARD_SIZE * CONV_FORWARD_SIZE;
	int poolPixs = POOL_FORWARD_SIZE * POOL_FORWARD_SIZE;
	
	const int numFilters = gridDim.y;
	const int imgIdx = blockIdx.x;
	const int filtIdx = blockIdx.y;

	if(threadIdx.x < POOL_FORWARD_SIZE && threadIdx.y < POOL_FORWARD_SIZE){
		out += imgIdx * numFilters * convPixs + filtIdx * convPixs \
				+ threadIdx.y * CONV_FORWARD_SIZE * MAX_POOL_Y \
				+ threadIdx.x * MAX_POOL_X; 
		dE_dy_i += imgIdx * numFilters * poolPixs + filtIdx * poolPixs \
				   + threadIdx.y * POOL_FORWARD_SIZE + threadIdx.x;
		maxPoolPos += imgIdx * numFilters * poolPixs + filtIdx * poolPixs \
				   + threadIdx.y * POOL_FORWARD_SIZE + threadIdx.x;
		int pos = maxPoolPos[0];
		int row = pos / MAX_POOL_Y;
		int col = pos % MAX_POOL_X;
		out[row * CONV_FORWARD_SIZE + col] = dE_dy_i[0];
	}
}

__global__ void compute_dE_db_h(const float* dE_dx_h, float* dE_db_h) {
	extern __shared__ float result[];
	
	const int idx = threadIdx.x + blockDim.x * threadIdx.y; 
	const int filtIdx = blockIdx.y;
	const int imgIdx = blockIdx.x;
	const int numFilters = gridDim.y;

	if(idx == 0)
		result[0] = 0;
	//某一张24*24的起始位置，本函数是将这24*24个点全部加起来最后生成一个点
	const int filPixs = CONV_FORWARD_SIZE * CONV_FORWARD_SIZE;
	dE_dx_h += imgIdx * numFilters * filPixs + filtIdx * filPixs + threadIdx.x;

	float ele = dE_dx_h[0];

	__syncthreads();
	atomicAdd(result, ele);
	__syncthreads();

	if (idx == 0) {
		dE_db_h[imgIdx * numFilters + filtIdx] = result[0] / filPixs;
	}
} 















