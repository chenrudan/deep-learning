/*
 * filename: nvmatrix_kernel.cuh
 */

#ifndef NVMATRIX_KERNEL_H_
#define NVMATRIX_KERNEL_H_

#define COPY_BLOCK_SIZE                     16
#define NUM_BLOCKS_MAX                      65535

#define DIVUP(a, b)                     (((a) + (b) - 1) / (b))

 __global__ void multiRowCol(float* aData, float* bData, float scaleAB, \
		         float* target, const int numInRowCol, const int times);

 __global__ void kAddColVector(float* mat, float* vec, float* tgtMat, \
         const unsigned int width, const unsigned int height, \
         const float scaleVec); 
 __global__ void kAddRowVector(float* mat, float* vec, float* tgtMat, \
         const unsigned int width, const unsigned int height, \
         const float scaleVec); 

__global__ void kSoftmax(float* gData, float* target, unsigned int numElements);

__global__ void kReciprocal(float* gData, float* target, unsigned int numElements);

__global__ void kLog(float* gData, float* target, unsigned int numElements);

__global__ void kDumbSumCols(float* mat, float* vec, unsigned int width, \
         unsigned int height); 

__global__ void kDumbSumRows(float* mat, float* vec, unsigned int width, \
	      unsigned int height);

__global__ void kSumRowInterval(float* mat, float* vec, unsigned int width, \
	     unsigned int height, int interval);


__global__ void kDumbMaxCols(float* mat, float* vec, unsigned int width, \
         unsigned int height); 

 __global__ void kMultByColVector(float* mat, float* vec, float* tgtMat, \
         unsigned int width, unsigned int height);

//__global__ void kSubtractFromScalar(float* gData, float scalar, float* target, \
//       unsigned int numElements);
__global__ void kSubtractFromScalar(float* gData, float* target, \
         unsigned int numElements);

 __global__ void kMult(float* matA, float* matB, float* tgtMat, \
         unsigned int width, unsigned int height);


__global__ void kAdd(float* matA, float* matB, float* tgtMat, float scaleA,  \
         float scaleB, unsigned int width, unsigned int height);


__global__ void kTranspose(float* srcData, float* dstData, \
		const int biggerDim, const int smallerDim, const int times);






#endif
