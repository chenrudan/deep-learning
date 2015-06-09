///
/// \file matrix_kernel.cuh
/// \brief matrix类的kernel函数

#ifndef MATRIX_KERNEL_H_
#define MATRIX_KERNEL_H_

#define NUM_BLOCKS_MAX                      65535

#define ADD_BLOCK_SIZE						16
#define COPY_BLOCK_SIZE                     16

#define DIVUP(a, b)                     (((a) + (b) - 1) / (b))


__global__ void kTranspose(float* srcData, float* dstData, \
		const int width, const int height);

/// \brief gpu实现addRowVector
///
/// \param[in] width 传递矩阵的长宽
__global__ void kAddRowVector(float* mat, float* vec, float* tgtMat, \
		int width, int height, float scaleVec); 

__global__ void kSubtractFromScalar(float* gData, float scalar, float* target, \
		unsigned int width, unsigned int height);

__global__ void kSoftmax(float* gData, unsigned int width, unsigned int height);

__global__ void kReciprocal(float* gData, float* target, unsigned int width, \
		unsigned int height);

__global__ void kLog(float* gData, float* target, unsigned int width, \
		unsigned int height);

__global__ void kSigmoid(float* gData, float* target, unsigned int width, \
		unsigned int height);

__global__ void kDumbSumCols(float* mat, float* vec, unsigned int width, \
		unsigned int height); 

__global__ void kDumbMaxPosInRow(float* mat, float* vec, unsigned int width, \
		unsigned int height); 

__global__ void kMult(float* matA, float* matB, float* tgtMat, \
		unsigned int width, unsigned int height);

__global__ void kAdd(float* matA, float* matB, float* tgtMat, float scaleA,  \
		float scaleB, unsigned int width, unsigned int height);








#endif
