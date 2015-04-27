/*
 * filename:nvmatrix_kernel.cu
 */

#include <cuda_runtime.h>
#include "nvmatrix_kernel.cuh"

__constant__ int WIDTH;
__constant__ int HEIGHT;
__constant__ float SCALE_VEC;


__device__ float mySigmoid(float x) {
	if(x < -300)
		return 0;
	else if( x > 300)
		return 1;
	else
		return 1 / (1 + __expf(-x));
}


__global__ void multiRowCol(float* aData, float* bData, float scaleAB, \
		float* target, const int numInRowCol, const int times ){
	extern __shared__ float result[];
	//a的每一行与b的每一行相乘

	const unsigned int idx = threadIdx.x * blockDim.y + threadIdx.y;
	const int threadNum = blockDim.x * blockDim.y;
	const int mIdx = blockIdx.x;
	const int nIdx = blockIdx.y;

	aData += mIdx * numInRowCol + idx;
	bData += nIdx * numInRowCol + idx;
	target += mIdx * gridDim.y + nIdx;

	if(idx == 0){
		result[0] = 0;
	}

	float ele = 0;
	for(int i = 0; i < times; i++){
		ele += scaleAB * aData[i * threadNum] * bData[i * threadNum];
	}
	if((threadNum * times < numInRowCol) && (idx < numInRowCol - threadNum * times)){
		ele += scaleAB * aData[threadNum * times] \
			   * bData[threadNum * times];
	}
	__syncthreads();
	atomicAdd(result, ele);
	__syncthreads();

	if(idx == 0){
		target[0] = result[0];
	}
}


__global__ void kAddRowVector(float* mat, float* vec, float* tgtMat, \
		unsigned int width, unsigned int height, float scaleVec) {

	const unsigned int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idx = idxY * width + idxX;
	const unsigned int numThreads = blockDim.x * gridDim.x * \
									blockDim.y * gridDim.y;

	//此处控制了线程数要小于行列积
	for (unsigned int i = idx; i < width * height; i += numThreads) {
		tgtMat[idx] = mat[idx] + scaleVec * vec[idx % width];

	}
}

__global__ void kSoftmax(float* gData, unsigned int width, \
		unsigned int height) {   

	//跟同一个block里面值比较大小取最大值，减去最大值
	const unsigned int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idx = idxY * width + idxX;
	//数据放入共享内存
	//计算离行值最近的2的次方
	int pow2Length = width;
	if(pow2Length & (pow2Length - 1)){
		while(pow2Length & (pow2Length - 1)){
			pow2Length &= pow2Length - 1;
		}
	}
	extern __shared__ float ori[];
	__shared__ float max;

	if(idxX < width)
		ori[idxX] = gData[idx];
	__syncthreads();

	//先通过reduce来求最大值
	if(idxX >= pow2Length && idxX < width)
		ori[idxX - pow2Length] = ori[idxX - pow2Length] > ori[idxX] \
								 ? ori[idxX - pow2Length] : ori[idxX];
	__syncthreads();

	for(int activeThreads = (pow2Length >> 1); activeThreads; activeThreads >>= 1){
		if(idxX < activeThreads){
			ori[idxX] = ori[idxX + activeThreads] > ori[idxX] \
						? ori[idxX + activeThreads] : ori[idxX];
		}
		__syncthreads();

	}
	if(idxX == 0)
		max = ori[0];
	__syncthreads();

	if(idxX < width)
		gData[idx] = __expf(gData[idx] - max);

	//reduce求和
	if(idxX < width)
		ori[idxX] = gData[idx];
	__syncthreads();

	if(idxX >= pow2Length && idxX < width)
		ori[idxX - pow2Length] += ori[idxX];
	__syncthreads();

	for(int activeThreads = (pow2Length >> 1); activeThreads; activeThreads >>= 1){
		if(idxX < activeThreads){
			ori[idxX] += ori[idxX + activeThreads];
		}
		__syncthreads();
	}

	if(idxX < width)
		gData[idx] = gData[idx] / ori[0];

}

__global__ void kSigmoid(float* gData, float* target, unsigned int width, \
		unsigned int height) {
	const unsigned int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idx = idxY * width + idxX;

	if(idxY < height && idxX < width)
		target[idx] = mySigmoid(gData[idx]);
}


__global__ void kReciprocal(float* gData, float* target, unsigned int width, \
		unsigned int height) {
	const unsigned int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idx = idxY * width + idxX;

	if(idxY < height && idxX < width)
		target[idx] = 1 / gData[idx];
}

__global__ void kLog(float* gData, float* target, unsigned int width, \
		unsigned int height) {   

	const unsigned int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idx = idxY * width + idxX;

	if(idxY < height && idxX < width){
		double tmp = gData[idx] < 1 - 10e-15 ? gData[idx] : 1 - 10e-15;
		tmp = tmp > 10e-15 ? tmp : 10e-15;
		target[idx] = __logf(gData[idx]);
	}
}

__global__ void kCompactCol(const float* ori, float* target, const int interval, \
		unsigned int width, unsigned int height){
	const unsigned int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int oriIdx = idxY * width * interval + idxX * interval;
	const unsigned int tarIdx = idxY * width + idxX;

	if(idxY < height && idxX < width){
		target[tarIdx] = 0;
		for(int i = 0; i < interval; i++){
			target[tarIdx] += ori[i + oriIdx];
		}
	}

}


__global__ void kDumbSumCols(float* mat, float* vec, unsigned int width, \
		unsigned int height) {

	const unsigned int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idx = idxY * width + idxX;

	extern __shared__ float ori[];

	int pow2Length = width;
	if(pow2Length & (pow2Length - 1)){
		while(pow2Length & (pow2Length - 1)){
			pow2Length &= pow2Length - 1;
		}
	}

	//reduce求和
	if(idxX < width)
		ori[idxX] = mat[idx];
	__syncthreads();

	if(idxX >= pow2Length && idxX < width)
		ori[idxX - pow2Length] += ori[idxX];
	__syncthreads();

	for(int activeThreads = (pow2Length >> 1); activeThreads; activeThreads >>= 1){ 
		if(idxX < activeThreads){
			ori[idxX] += ori[idxX + activeThreads];
		}
		__syncthreads();
	}   

	if(idxX == 0)
		vec[idxY] = ori[0];

}


__global__ void kDumbMaxPosInRow(float* mat, float* vec, unsigned int width, \
		unsigned int height) {
	const unsigned int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idx = idxY * width + idxX;

	extern __shared__ float ori[];

	int pow2Length = width;
	if(pow2Length & (pow2Length - 1)){
		while(pow2Length & (pow2Length - 1)){
			pow2Length &= pow2Length - 1;
		}
	}

	//reduce求最大值
	if(idxX < width)
		ori[idxX] = mat[idx];
	__syncthreads();

	if(idxX >= pow2Length && idxX < width)
		ori[idxX - pow2Length] = ori[idxX - pow2Length] > ori[idxX] \
								 ? ori[idxX - pow2Length] : ori[idxX];
	__syncthreads();

	for(int activeThreads = (pow2Length >> 1); activeThreads; activeThreads >>= 1){ 
		if(idxX < activeThreads){
			ori[idxX] = ori[idxX + activeThreads] > ori[idxX] \
						? ori[idxX + activeThreads] : ori[idxX];
		}
		__syncthreads();
	}   

	if(mat[idx] == ori[0] && idxX < width)
		vec[idxY] = idxX;

	__syncthreads();
}

__global__ void kMultByColVector(float* mat, float* vec, float* tgtMat, \
		unsigned int width, unsigned int height) {
	const unsigned int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idx = idxY * width + idxX;

	if(idxY < height && idxX < width)
		tgtMat[idx] = mat[idx] * vec[idxY];
}

__global__ void kSubtractFromScalar(float* gData, float scalar, float* target, \
		unsigned int width, unsigned int height) {
	const unsigned int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idx = idxY * width + idxX;

	if(idxY < height && idxX < width)
		target[idx] = scalar - gData[idx];
}

__global__ void kMult(float* matA, float* matB, float* tgtMat, \
		unsigned int width, unsigned int height) {
	const unsigned int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idx = idxY * width + idxX;

	if(idxY < height && idxX < width)
		tgtMat[idx] = matA[idx] * matB[idx];
}

__global__ void kAdd(float* matA, float* matB, float* tgtMat, float scaleA,  \
		float scaleB, unsigned int width, unsigned int height) {
	const unsigned int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idx = idxY * width + idxX;

	if(idxY < height && idxX < width)
		tgtMat[idx] = scaleA * matA[idx] + scaleB * matB[idx];
}


__global__ void kTranspose(float* srcData, float* dstData, \
		const int width, const int height){

	const unsigned int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int srcIdx = idxY * width + idxX;
	const unsigned int dstIdx = idxX * height + idxY;

	if(idxY < height && idxX < width)
		dstData[dstIdx] = srcData[srcIdx];

}




