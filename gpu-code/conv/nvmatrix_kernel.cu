/*
 * filename:nvmatrix_kernel.cu
 */

#include <cuda_runtime.h>
#include "nvmatrix_kernel.cuh"

__constant__ float dMinus = 1;

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


__global__ void kAddColVector(float* mat, float* vec, float* tgtMat, \
		const unsigned int width, const unsigned int height, \
		const float scaleVec) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numThreads = blockDim.x * gridDim.x;

	for (unsigned int i = idx; i < width * height; i += numThreads) {
		tgtMat[i] = mat[i] + scaleVec * vec[i / width];
	}
}

__global__ void kAddRowVector(float* mat, float* vec, float* tgtMat, \
		const unsigned int width, const unsigned int height, \
		const float scaleVec) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numThreads = blockDim.x * gridDim.x;

//	for (unsigned int i = idx; i < width * height; i += numThreads) {
		tgtMat[idx] = mat[idx] + scaleVec * vec[idx % width];
		
//	}
}

__global__ void kSoftmax(float* gData, float* target, unsigned int numCols) {   

	//跟同一个block里面值比较大小取最大值，减去最大值

//	const unsigned int idx = blockIdx.x * numCols + threadIdx.x;
	gData += blockIdx.x * numCols;
	target += blockIdx.x * numCols;

	double max = gData[0];
	for (unsigned int i = 1; i < numCols; i++){
		if(max < gData[i])
			max = gData[i];
	}
	double sum = 0;
	for (unsigned int i = 0; i < numCols; i++){
		target[i] = __expf(gData[i] - max);
		sum += target[i];
	}
	for (unsigned int i = 0; i < numCols; i++){
		target[i] = target[i] / sum;
	}
}

__global__ void kReciprocal(float* gData, float* target, unsigned int numElements) {

	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
		target[i] = 1 / gData[i];
}

__global__ void kLog(float* gData, float* target, unsigned int numElements) {   

	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x){
		double tmp = gData[i] < 1 - 10e-15 ? gData[i] : 1 - 10e-15;
		tmp = tmp > 10e-15 ? tmp : 10e-15;
		target[i] = __logf(gData[i]);
	}
}

__global__ void kDumbSumCols(float* mat, float* vec, unsigned int width, \
		unsigned int height) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < height) {
		mat += idx * width;
		float sum = 0;
		for (int j = 0; j < width; j++) {
			sum += mat[j];
		}
		vec[idx] = sum;
	}
}

__global__ void kDumbSumRows(float* mat, float* vec, unsigned int width, \
		unsigned int height) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < width) {
		mat += idx;
		float sum = 0;
		for (int j = 0; j < height; j++) {
			sum += mat[j * width];
		}
		vec[idx] = sum;
	}
}

__global__ void kSumRowInterval(float* mat, float* vec, unsigned int width, \
		unsigned int height, int interval) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < height) {
		mat += idx * width;
		float sum = 0;
		for (int j = 0; j < width; j += interval) {
			sum += mat[j];
		}
		vec[idx] = sum;
	}
}

__global__ void kDumbMaxCols(float* mat, float* vec, unsigned int width, \
		unsigned int height) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < height) {
		mat += idx * width;
		float mx = mat[0];
		for (int j = 1; j < width; j++) {
			mx = mat[j] > mx ? mat[j] : mx;
		}
		vec[idx] = mx;
	}
}

__global__ void kMultByColVector(float* mat, float* vec, float* tgtMat, \
		unsigned int width, unsigned int height) {
	//block.x表示行数，threadIdx.x表示列数
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numThreads = blockDim.x * gridDim.x;

	tgtMat[idx] = mat[idx] * vec[blockIdx.x];


//	for (unsigned int i = idx; i < width * height; i += numThreads) {
//		tgtMat[i] = mat[i] * vec[i / width];
//	}
}

//__global__ void kSubtractFromScalar(float* gData, float scalar, float* target, \
		unsigned int numElements) {
__global__ void kSubtractFromScalar(float* gData, float* target, \
		unsigned int numElements) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
		target[i] = 1 - gData[i];
}

__global__ void kMult(float* matA, float* matB, float* tgtMat, \
		unsigned int width, unsigned int height) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numThreads = blockDim.x * gridDim.x;

	for (unsigned int i = idx; i < width * height; i += numThreads) {
		tgtMat[i] = matA[i] * matB[i];
	}
}

__global__ void kAdd(float* matA, float* matB, float* tgtMat, float scaleA,  \
		float scaleB, unsigned int width, unsigned int height) {
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int numThreads = blockDim.x * gridDim.x;

	for (unsigned int i = idx; i < width * height; i += numThreads) {
		tgtMat[i] = scaleA * matA[i] + scaleB * matB[i];
	}
}


__global__ void kTranspose(float* srcData, float* dstData, \
		const int biggerDim, const int row, const int times){

	const unsigned int idx = threadIdx.x;
	//mIdx代表维数较小的一方，nIdx代表维数较大的一方
	const unsigned int mIdx = blockIdx.x;
	const unsigned int smallerDim = gridDim.x;

	for(int i = 0; i < times; i++){
		const unsigned int nIdx = i * blockDim.x + threadIdx.x;
		//假如行小于列，那么转置后的行大于列
		if(smallerDim == row)
			dstData[nIdx * smallerDim + mIdx] = srcData[mIdx * biggerDim + nIdx];
		else
			dstData[mIdx * biggerDim + nIdx] = srcData[nIdx * smallerDim + mIdx];

	}
	if(idx < biggerDim - blockDim.x * times){
		const unsigned int nIdx = times * blockDim.x + threadIdx.x;
		if(smallerDim == row)
			dstData[nIdx * smallerDim + mIdx] = srcData[mIdx * biggerDim + nIdx];
		else
			dstData[mIdx * biggerDim + nIdx] = srcData[nIdx * smallerDim + mIdx];
	}
}




