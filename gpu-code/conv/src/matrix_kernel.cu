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
__global__ void kAddRowVector(Dtype* mat, Dtype* vec, Dtype* tgtMat, \
		const int width, const int height, float scaleVec) {

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
__global__ void kSoftmax(Dtype* gData, Dtype* target, const int width, \
		const int height) {   

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
__global__ void kRelu(Dtype* gData, Dtype* target, int* record, const int width, \
		const int height) {
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
__global__ void kReluBack(Dtype* gData, Dtype* target, int* record, const int width, \
		const int height) {
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
__global__ void kSigmoid(Dtype* gData, Dtype* target, const int width, \
		const int height) {
	const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const int idx = idxY * width + idxX;

	if(idxY < height && idxX < width)
		target[idx] = mySigmoid(gData[idx]);
}


template <typename Dtype>
__global__ void kReciprocal(Dtype* gData, Dtype* target, const int width, \
		const int height) {
	const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const int idx = idxY * width + idxX;

	if(idxY < height && idxX < width)
		target[idx] = 1 / gData[idx];
}

template <typename Dtype>
__global__ void kLog(Dtype* gData, Dtype* target, const int width, \
		const int height) {   

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
		const int width, const int height){
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
__global__ void kDumbSumCols(Dtype* mat, Dtype* vec, const int width, \
		const int height) {

	extern __shared__ Dtype ori[];

	//距离width最近的2次幂
	int pow2Length = width;
	if(pow2Length & (pow2Length - 1)){
		while(pow2Length & (pow2Length - 1)){
			pow2Length &= pow2Length - 1;
		}
	}
	

	//reduce求和
	int i = threadIdx.x;
	while(i < width){
		ori[i] = mat[blockIdx.x * width + i];
		i += blockDim.x;
	}
	__syncthreads();
	int reduce_len = pow2Length > blockDim.x ? blockDim.x : pow2Length;

	//需要执行reduce的次数，一次性只能执行最多32*32
	int times = width / reduce_len;

	//把最后无法整除的地方先处理
	int idxX = threadIdx.x + reduce_len * times;
	if(idxX > (reduce_len * times) && idxX < width)
		ori[idxX - reduce_len] += ori[idxX];
	__syncthreads();


	for(int j = times - 1; j >= 0; j--){
		idxX = threadIdx.x + j * reduce_len;
		if(threadIdx.x == 0 && ((j + 1) * reduce_len) < width)
			ori[0] += ori[(j + 1) * reduce_len];
		__syncthreads();
		for(int activeThreads = (reduce_len >> 1); activeThreads; activeThreads >>= 1){ 
			if(threadIdx.x < activeThreads){
				ori[idxX] += ori[idxX + activeThreads];
			}
			__syncthreads();
		}
	}

	if(threadIdx.x == 0){
		vec[blockIdx.x] = ori[0];
	}
	__syncthreads();

}


template <typename Dtype>
__global__ void kDumbMaxPosInRow(Dtype* mat, Dtype* vec, const int width, \
		const int height) {
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
		const int width, const int height) {
	const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const int idx = idxY * width + idxX;

	if(idxY < height && idxX < width)
		tgtMat[idx] = mat[idx] * vec[idxY];
}

template <typename Dtype>
__global__ void kSubtractFromScalar(Dtype* gData, float scalar, Dtype* target, \
		const int width, const int height) {
	const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const int idx = idxY * width + idxX;

	if(idxY < height && idxX < width)
		target[idx] = scalar - gData[idx];
}

template <typename Dtype>
__global__ void kMult(Dtype* matA, Dtype* matB, Dtype* tgtMat, \
		const int width, const int height) {
	const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const int idx = idxY * width + idxX;

	if(idxY < height && idxX < width)
		tgtMat[idx] = matA[idx] * matB[idx];
}

template <typename Dtype>
__global__ void kAdd(Dtype* matA, Dtype* matB, Dtype* tgtMat, float scaleA,  \
		float scaleB, const int width, const int height) {
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

template <typename Dtype>
__global__ void kComputeNorm(const Dtype* vec, Dtype* norm, const int len){
	//每一个block计算一个模
	extern __shared__ Dtype sh_norm[];

	int pow2_len = len;
	if (pow2_len & (pow2_len - 1)) {
		while (pow2_len & (pow2_len - 1)){
			pow2_len &= pow2_len - 1;
		}
	}

	int i = threadIdx.x;
	while (i < len) {
		sh_norm[i] = vec[i]*vec[i];
		i += blockDim.x;
	}

	int reduce_len = pow2_len > blockDim.x ? blockDim.x : pow2_len;
	int times = len / reduce_len;

	int vec_pos = threadIdx.x + reduce_len * times;
	if (vec_pos > (reduce_len * times) && vec_pos < len) {
		sh_norm[vec_pos - reduce_len] += sh_norm[vec_pos];
	}
	__syncthreads();

	for (int j = times-1; j >= 0; j--) {
		vec_pos = threadIdx.x + j*reduce_len;
		if (threadIdx.x == 0 && (j + 1) * reduce_len < len) {
			sh_norm[0] += sh_norm[(j + 1) * reduce_len];
		}
		__syncthreads();
		for (int active_thread = (reduce_len >> 1); active_thread; active_thread >>= 1) {
			if (threadIdx.x < active_thread) {
				sh_norm[vec_pos] += sh_norm[vec_pos + active_thread];
			}
			__syncthreads();
		}
	}

	if (threadIdx.x == 0) {
		norm[0] = sqrt(sh_norm[0]);
	}

	__syncthreads();
}

template <typename Dtype>
__global__ void kCropImg(const Dtype* ori_img, Dtype* dst_img, \
		const int row_start, const int cropped_height, \
		const int col_start, const int cropped_width, \
		const int ori_width){

	int idx = threadIdx.x;

	while (idx < cropped_height*cropped_width) {
		int ori_row_idx = idx / cropped_width + row_start;
		int ori_col_idx = idx % cropped_width + col_start;
		dst_img[idx] = ori_img[ori_row_idx*ori_width + ori_col_idx];
		idx += blockDim.x;
	}
	__syncthreads();
}

template <typename Dtype>
__global__ void kComputeHouseholderVec(const Dtype* src, Dtype* dst, \
		Dtype added_value, Dtype scale, const int len) {
	int idx = threadIdx.x;
	while (idx < len) {
		if (idx == 0) {
			dst[idx] = scale * (src[idx] + added_value);
		} else
			dst[idx] = scale * src[idx];
		idx += blockDim.x;
	}
}

template <typename Dtype>
__global__ void kSubedByUnitMat(Dtype* matA, Dtype* tgtMat, \
		const int width, const int height) {
	const int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	const int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	const int idx = idxY * width + idxX;

	if(idxY < height && idxX < width ){
		if ( idxX == idxY)
			tgtMat[idx] = 1 - matA[idx];
		else
			tgtMat[idx] = - matA[idx];
	}

}