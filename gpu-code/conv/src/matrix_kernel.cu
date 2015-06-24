/*
 * filename:nvmatrix_kernel.cu
 */

#include <cuda_runtime.h>
#include "matrix_kernel.hpp"

template <typename Dtype>
__device__ Dtype mySigmoid(Dtype x) {
	if(x < -300)
		return 0;
	else if( x > 300)
		return 1;
	else
		return 1 / (1 + __expf(-x));
}

template <typename Dtype>
__global__ void multiRowCol(Dtype* aData, Dtype* bData, float scaleAB, \
		Dtype* target, const int numInRowCol, const int times ){
	extern __shared__ Dtype result[];
	//a的每一行与b的每一行相乘

	const int idx = threadIdx.x * blockDim.y + threadIdx.y;
	const int threadNum = blockDim.x * blockDim.y;
	const int mIdx = blockIdx.x;
	const int nIdx = blockIdx.y;

	aData += mIdx * numInRowCol + idx;
	bData += nIdx * numInRowCol + idx;
	target += mIdx * gridDim.y + nIdx;

	if(idx == 0){
		result[0] = 0;
	}

	Dtype ele = 0;
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


template <typename Dtype>
__global__ void kAddRowVector(Dtype* mat, Dtype* vec, Dtype* tgtMat, \
		int width, int height, float scaleVec) {

	const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const int idx = idxY * width + idxX;
	const int numThreads = blockDim.x * gridDim.x * \
									blockDim.y * gridDim.y;

	//此处控制了线程数要小于行列积
	for (int i = idx; i < width * height; i += numThreads) {
		tgtMat[idx] = mat[idx] + scaleVec * vec[idx % width];

	}
}

template <typename Dtype>
__global__ void kSoftmax(Dtype* gData, Dtype* target, int width, \
		int height) {   

	//跟同一个block里面值比较大小取最大值，减去最大值
	const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const int idx = idxY * width + idxX;
	//数据放入共享内存
	//计算离行值最近的2的次方
	int pow2Length = width;
	if(pow2Length & (pow2Length - 1)){
		while(pow2Length & (pow2Length - 1)){
			pow2Length &= pow2Length - 1;
		}
	}
	extern __shared__ Dtype ori[];
	__shared__ Dtype max;

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
		target[idx] = __expf(gData[idx] - max);

	//reduce求和
	if(idxX < width)
		ori[idxX] = target[idx];
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
		target[idx] = target[idx] / ori[0];

}

template <typename Dtype>
__global__ void kRelu(Dtype* gData, Dtype* target, int* record, int width, \
		int height) {
	const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const int idx = idxY * width + idxX;

	if(idxY < height && idxX < width){
		if(gData[idx] > 0){
			target[idx] = gData[idx];
			record[idx] = 1;
		}else{
			target[idx] = 0;
			record[idx] = 0;
		}
	}
}
template <typename Dtype>
__global__ void kReluBack(Dtype* gData, Dtype* target, int* record, int width, \
		int height) {
	const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const int idx = idxY * width + idxX;

	if(idxY < height && idxX < width){
		if(record[idx] == 1){
			target[idx] = gData[idx];
		}else{
			target[idx] = 0;
		}
	}
}

template <typename Dtype>
__global__ void kSigmoid(Dtype* gData, Dtype* target, int width, \
		int height) {
	const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const int idx = idxY * width + idxX;

	if(idxY < height && idxX < width)
		target[idx] = mySigmoid(gData[idx]);
}


template <typename Dtype>
__global__ void kReciprocal(Dtype* gData, Dtype* target, int width, \
		int height) {
	const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const int idx = idxY * width + idxX;

	if(idxY < height && idxX < width)
		target[idx] = 1 / gData[idx];
}

template <typename Dtype>
__global__ void kLog(Dtype* gData, Dtype* target, int width, \
		int height) {   

	const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const int idx = idxY * width + idxX;

	if(idxY < height && idxX < width){
		double tmp = gData[idx] < 1 - 10e-15 ? gData[idx] : 1 - 10e-15;
		tmp = tmp > 10e-15 ? tmp : 10e-15;
		target[idx] = __logf(gData[idx]);
	}
}

template <typename Dtype>
__global__ void kCompactCol(const Dtype* ori, Dtype* target, const int interval, \
		int width, int height){
	const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const int oriIdx = idxY * width * interval + idxX * interval;
	const int tarIdx = idxY * width + idxX;

	if(idxY < height && idxX < width){
		target[tarIdx] = 0;
		for(int i = 0; i < interval; i++){
			target[tarIdx] += ori[i + oriIdx];
		}
	}

}


template <typename Dtype>
__global__ void kDumbSumCols(Dtype* mat, Dtype* vec, int width, \
		int height) {

	const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const int idx = idxY * width + idxX;

	extern __shared__ Dtype ori[];

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


template <typename Dtype>
__global__ void kDumbMaxPosInRow(Dtype* mat, Dtype* vec, int width, \
		int height) {
	const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const int idx = idxY * width + idxX;

	extern __shared__ Dtype ori[];

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

template <typename Dtype>
__global__ void kMultByColVector(Dtype* mat, Dtype* vec, Dtype* tgtMat, \
		int width, int height) {
	const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const int idx = idxY * width + idxX;

	if(idxY < height && idxX < width)
		tgtMat[idx] = mat[idx] * vec[idxY];
}

template <typename Dtype>
__global__ void kSubtractFromScalar(Dtype* gData, float scalar, Dtype* target, \
		int width, int height) {
	const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const int idx = idxY * width + idxX;

	if(idxY < height && idxX < width)
		target[idx] = scalar - gData[idx];
}

template <typename Dtype>
__global__ void kMult(Dtype* matA, Dtype* matB, Dtype* tgtMat, \
		int width, int height) {
	const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const int idx = idxY * width + idxX;

	if(idxY < height && idxX < width)
		tgtMat[idx] = matA[idx] * matB[idx];
}

template <typename Dtype>
__global__ void kAdd(Dtype* matA, Dtype* matB, Dtype* tgtMat, float scaleA,  \
		float scaleB, int width, int height) {
	const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const int idx = idxY * width + idxX;

	if(idxY < height && idxX < width)
		tgtMat[idx] = scaleA * matA[idx] + scaleB * matB[idx];
}


template <typename Dtype>
__global__ void kTranspose(Dtype* srcData, Dtype* dstData, \
		const int width, const int height){

	const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const int srcIdx = idxY * width + idxX;
	const int dstIdx = idxX * height + idxY;

	if(idxY < height && idxX < width)
		dstData[dstIdx] = srcData[srcIdx];

}




