/*
 * filename:test.cu
 */

#include <iostream>
#include <time.h>
#include "cublas_v2.h"

#include "matrix.h"
#include "nvmatrix.cuh"
#include "convnet.cuh"
#include "convnet_kernel.cuh"

using namespace std;

__global__ void add(float *a, float *b, float *c, int length){
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < length)
		c[idx] = a[idx] + b[idx];
}


int main(){
	int m = 100;
	int k = 784;
	int n = 10;
	cublasHandle_t handle;
	cublasCreate(&handle);

	NVMatrix* a = new NVMatrix(100, 784);
	NVMatrix* b = new NVMatrix(100, 24*24*16);
	NVMatrix* c = new NVMatrix(25, 100*24*24);
	NVMatrix* d = new NVMatrix(25, 16);
	NVMatrix* e = new NVMatrix(100*24*24, 16);
	NVMatrix* f = new NVMatrix(16, 25);

	a->reValue(28);
	b->reValue(24);

        int numKernels = 100*24*24*5*5*1;;
        int numBlocks = numKernels / 1024 + 1;

	//im2col_conv<<<numBlocks, 1024>>>(a->getDevData(), \
			c->getDevData(), numKernels, 24*24, \
			100*24*24, 25);

	a->showValue("a");
	c->showValue("c");

	numKernels = 100*24*24*16;
	numBlocks = numKernels / 1024;
	reshape_dE_dx_h<<<numBlocks, 1024>>>(e->getDevData(), b->getDevData(), \
				numKernels);

//cout << c->getNumRows() << ":" << c->getNumCols() << endl;
//cout << d->getNumRows() << ":" << d->getNumCols() << endl;
	c->rightMult(e, 1, d, handle);
	d->getTranspose(f);

//	numKernels = 100*24*24*16;
//	numBlocks = numKernels / 1024;
//	reshape_y_h<<<numBlocks, 1024>>>(e->getDevData(), f->getDevData(), \
			numKernels);


	b->showValue("b");
	e->showValue("e");
	f->showValue("f");

	//	a->showValue("a");

	//	a->apply(NVMatrix::SOFTMAX, 1, 1);
	//	a->addRowVector(b, 16 ,10);
	//a->eltWiseMult(b, c, 16, 16);
	//	a->rightMult(b, 1, c, handle);
	//	a->getTranspose(b);
	//	a->sumRow(b,1,1);
	//	a->maxPosInRow(b);
	//	a->eltWiseMultByColVector(b);
	/*	a->showValue("a");
		a->showValue("a");
		NVMatrix* d = a->sumCol(16, 10);
		d->showValue("d");
	 */
	//	a->eltWiseDivideByVector(b, 16, 10);
	//	a->sumCol(b, 16, 10);
	//	a->add(b, 1, 1, 16, 10);

	cout << "done1\n";
	cout << "done2\n";
	//	c->showValue("c");

	cublasDestroy(handle);

	return 0;
}
