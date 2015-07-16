///
/// \file matrix_kernel.cuh
/// \brief matrix类的kernel函数

#ifndef MATRIX_KERNEL_H_
#define MATRIX_KERNEL_H_

#define NUM_BLOCKS_MAX                      65535

#define ADD_BLOCK_SIZE						16
#define COPY_BLOCK_SIZE                     16

#define DIVUP(a, b)                     (((a) + (b) - 1) / (b))

template <typename Dtype>
__global__ void kTranspose(Dtype* srcData, Dtype* dstData, \
		const int width, const int height);

/// \brief gpu实现addRowVector
///
/// \param[in] width 传递矩阵的长宽
template <typename Dtype>
__global__ void kAddRowVector(Dtype* mat, Dtype* vec, Dtype* tgtMat, \
		const int width, const int height, float scaleVec); 

template <typename Dtype>
__global__ void kSubtractFromScalar(Dtype* gData, float scalar, Dtype* target, \
		const int width, const int height);

template <typename Dtype>
__global__ void kSoftmax(Dtype* gData, Dtype* target, const int width, \
		const int height);

template <typename Dtype>
__global__ void kReciprocal(Dtype* gData, Dtype* target, const int width, \
		const int height);

template <typename Dtype>
__global__ void kLog(Dtype* gData, Dtype* target, const int width, \
		const int height);

template <typename Dtype>
__global__ void kSigmoid(Dtype* gData, Dtype* target, const int width, \
		const int height);

template <typename Dtype>
__global__ void kRelu(Dtype* gData, Dtype* target, int* record, const int width, \
		const int height);

template <typename Dtype>
__global__ void kReluBack(Dtype* gData, Dtype* target, int* record, const int width, \
		const int height);

template <typename Dtype>
__global__ void kDumbSumCols(Dtype* mat, Dtype* vec, const int width, \
		const int height); 

template <typename Dtype>
__global__ void kDumbMaxPosInRow(Dtype* mat, Dtype* vec, const int width, \
		const int height); 

template <typename Dtype>
__global__ void kMult(Dtype* matA, Dtype* matB, Dtype* tgtMat, \
		const int width, const int height);

template <typename Dtype>
__global__ void kAdd(Dtype* matA, Dtype* matB, Dtype* tgtMat, float scaleA,  \
		float scaleB, const int width, const int height);

//dst = (src + [added_value, 0, ..., 0]) * scale
template <typename Dtype>
__global__ void kComputeHouseholderVec(const Dtype* src, Dtype* dst, \
		Dtype added_value, Dtype scale, const int len);

template <typename Dtype>
__global__ void kSubedByUnitMat(Dtype* matA, Dtype* tgtMat, \
		const int width, const int height);

#include "../src/matrix_kernel.cu"


#endif
