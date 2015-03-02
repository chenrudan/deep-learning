/*
 * filename:test.cu
 */

#include <iostream>
#include <time.h>
#include "cublas_v2.h"

#include "matrix.h"
#include "nvmatrix.cuh"

using namespace std;

__global__ void add(float *a, float *b, float *c, int length){
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < length)
		c[idx] = a[idx] + b[idx];
}


int main(){
	int m = 16;
	int k = 10;
	int n = 10;
	cublasHandle_t handle;
	cublasCreate(&handle);
	NVMatrix* a = new NVMatrix(m, k);
	NVMatrix* b = new NVMatrix(1, k);
	NVMatrix* c = new NVMatrix(m, n);

	a->reValue(28);
	b->reValue(20);
	a->showValue("a");
	
	a->addRowVector(b, 16 ,10);
	//a->eltWiseMult(b, c, 16, 16);
//	a->rightMult(b, 1, c, handle);
//	a->getTranspose(b);
/*	a->showValue("a");
	a->apply(NVMatrix::EXP, 16, 1);
	a->showValue("a");
	NVMatrix* d = a->sumCol(16, 10);
	d->showValue("d");
*/
	a->showValue("a");
//	a->eltWiseDivideByVector(b, 16, 10);
//	a->sumCol(b, 16, 10);
//	a->add(b, 1, 1, 16, 10);
	
	b->showValue("b");
	cout << "done1\n";
	cout << "done2\n";
//	c->showValue("c");

	cublasDestroy(handle);

	return 0;
}
