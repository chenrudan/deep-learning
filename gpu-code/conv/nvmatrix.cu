/*
 * filename: nvmatrix.cu
 */
#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include "cublas_v2.h"

#include "nvmatrix.cuh"
#include "nvmatrix_kernel.cuh"
#include "matrix.h"

using namespace std;

NVMatrix::NVMatrix(){
	_init(0, 0);
}
NVMatrix::NVMatrix(int numRows, int numCols){
	_init(numRows, numCols);
}
NVMatrix::NVMatrix(const Matrix* like, bool copy){
	_init(like->getNumRows(), like->getNumCols());
	if (copy) {
		copyFromHost(like);
	}
}
NVMatrix::NVMatrix(const NVMatrix* like, bool copy){
	_init(like->getNumRows(), like->getNumCols());
	if (copy) {
		copyFromDevice(like);
	}
}
// not copy the data
NVMatrix::NVMatrix(const Matrix* like) {
	_init(like->getNumRows(), like->getNumCols());
}
NVMatrix::NVMatrix(const NVMatrix* like) {
	_init(like->getNumRows(), like->getNumCols());
}
NVMatrix::NVMatrix(float* devData, int numRows, int numCols) {
	_numRows = numRows;
	_numCols = numCols;
	_numElements = numRows * numCols;
	_ownsData = false;
	_devData = devData;				
}

NVMatrix::~NVMatrix(){
	if(_ownsData && _numElements > 0){
		cudaFree(_devData);
	}
}

void NVMatrix::_init(unsigned int numRows, unsigned int numCols) {
	_numRows = numRows;
	_numCols = numCols;
	_numElements = numRows * numCols;
	_ownsData = true;
	if (_numElements > 0) {
		cudaError_t status = cudaMalloc((void**) &_devData, \
				_numRows * _numCols * sizeof(float));
		if (status != cudaSuccess) {
			fprintf(stderr, "!!!! device memory allocation error\n");
			exit(EXIT_FAILURE);
		}
	} 
}

void NVMatrix::copyFromHost(const Matrix* hostMatrix, bool resizeDeviceMatrix) {
	if(resizeDeviceMatrix) {
		resize(hostMatrix);
	}
	copyFromHost(hostMatrix);
}
void NVMatrix::copyFromHost(const Matrix* hostMatrix) {
	assert(isSameDims(hostMatrix));
	cudaError_t status = cudaMemcpy(_devData, hostMatrix->getData(), \
			sizeof(float) * hostMatrix->getNumRows() * hostMatrix->getNumCols(), \
			cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		cout << stderr, "!!!! device access error (write)\n";
		exit( EXIT_FAILURE);
	}
}  
void NVMatrix::copyFromHost(float* data, const int dataLength){
	cudaError_t status = cudaMemcpy(_devData, data, \
			sizeof(float) * dataLength, cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		cout << stderr, "!!!! device access error (write)\n";
		exit( EXIT_FAILURE);
	}
}
void NVMatrix::copyFromDevice(const NVMatrix* devMatrix) {
	assert(isSameDims(devMatrix));
	cudaError_t status = cudaMemcpy(_devData, devMatrix->getDevData(), \
			sizeof(float) * devMatrix->getNumRows() * devMatrix->getNumCols(), \
			cudaMemcpyDeviceToDevice);
	if (status != cudaSuccess) {
		cout << stderr, "!!!! device access error (write)\n";
		exit( EXIT_FAILURE);
	}
}
void NVMatrix::copyFromDevice(const NVMatrix* devMatrix, const int startPos) {
	cudaError_t status = cudaMemcpy(_devData, devMatrix->getDevData() + startPos, \
			sizeof(float) * _numRows * _numCols, \
			cudaMemcpyDeviceToDevice);
	if (status != cudaSuccess) {
		cout << stderr, "!!!! device access error (write)\n";
		exit( EXIT_FAILURE);
	}
}
void NVMatrix::copyToHost(Matrix* hostMatrix) const{
	assert(isSameDims(hostMatrix));
	cudaError_t status = cudaMemcpy(hostMatrix->getData(), _devData, \
			sizeof(float) * hostMatrix->getNumRows() * hostMatrix->getNumCols(), \
			cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		cout << stderr, "!!!! device access error (write)\n";
		exit( EXIT_FAILURE);
	}
}  

void NVMatrix::copyToHost(float* data, const int dataLength) const{
	cudaError_t status = cudaMemcpy(data, _devData, \
			sizeof(float) * dataLength, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		cout << stderr, "!!!! device access error (write)\n";
		exit( EXIT_FAILURE);
	}
}

void NVMatrix::zeros(){
	cudaMemset(_devData, 0, _numRows * _numCols * sizeof(float));
}

bool NVMatrix::resize(int numRows, int numCols){
	bool reallocated = false;
	if (numRows != _numRows || numCols != _numCols) {
		assert(_ownsData);
		if (_numElements != numRows * numCols) {
			cudaError_t status = cudaFree(_devData);
			if (status != cudaSuccess) {
				fprintf(stderr, "!!!! memory free error\n");
				exit(EXIT_FAILURE);
			}
			status = cudaMalloc((void **)&_devData, \
					numCols * numRows * sizeof(float));
			if (status != cudaSuccess) {
				fprintf(stderr, "!!!! device memory allocation error\n");
				exit(EXIT_FAILURE);
			}
			reallocated = true;
		}
		_numRows = numRows;
		_numCols = numCols;
		_numElements = numRows * numCols;
	}
	return reallocated;
}
bool NVMatrix::resize(const NVMatrix* like) {
	bool r = resize(like->getNumRows(), like->getNumCols());
	return r;
}

bool NVMatrix::resize(const Matrix* like) {
	bool r = resize(like->getNumRows(), like->getNumCols());
	return r;
}

void NVMatrix::checkBounds(int startRow, int endRow, int startCol, \
		int endCol) const {
	assert(startRow >= 0 && startRow <= _numRows);
	assert(endRow >= 0 && endRow <= _numRows);
	assert(startCol >= 0 && startCol <= _numCols);
	assert(endCol >= 0 && endCol <= _numCols);
}

NVMatrix* NVMatrix::getTranspose() {
	Matrix* ori = new Matrix(_numRows, _numCols);
	this->copyToHost(ori);
	Matrix* trans = ori->getTranspose();
	NVMatrix* nvTrans = new NVMatrix(trans, true);
	delete ori;
	delete trans;
	return nvTrans;
}

void NVMatrix::getTranspose(NVMatrix* target){
	
	int numBlocks, numThreadsPerBlock;
	int dimThreadsTimes;
	int biggerDim = _numRows > _numCols ? _numRows : _numCols;
	int smallerDim = _numRows < _numCols ? _numRows : _numCols;

	numBlocks = smallerDim;
	dimThreadsTimes = biggerDim / 1024;
	if(dimThreadsTimes < 1)
		numThreadsPerBlock = biggerDim;
	else
		numThreadsPerBlock = 1024;
	kTranspose<<<numBlocks, numThreadsPerBlock>>>(_devData, \
				target->getDevData(), biggerDim, _numRows, dimThreadsTimes);

}

void NVMatrix::rightMult(NVMatrix* b, float scaleAB, \
		NVMatrix *target, cublasHandle_t& handle) {

	clock_t t = clock();

	int m = this->_numRows;
	int k = this->_numCols;
	int n = b->getNumCols();
	float scaleTar = 0;
	assert(k == b->getNumRows());
//	if(m >= 256 || n >= 256){

		//列主
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &scaleAB, \
				b->getDevData(), n, this->getDevData(), k, \
				&scaleTar, target->getDevData(), n);
//	t = clock() - t;                                                                                        
//	cout << "sgemm: " << (float)t/CLOCKS_PER_SEC << " seconds. \n";
//	t = clock(); 
	/*
	}
	else{

		dim3 blocks = dim3(m, n);
		dim3 threads;
		int eachThreadComputeTime = k / 1024;
		if(eachThreadComputeTime <= 1)
			threads = dim3(16, k / 16);
		else
			threads = dim3(16, 64);


		NVMatrix *bTrans = new NVMatrix(b->getNumCols(), b->getNumRows());
		b->getTranspose(bTrans);
	

		multiRowCol<<<blocks, threads, sizeof(float)>>>(this->_devData, \
				bTrans->getDevData(), scaleAB, target->getDevData(), \
				k, eachThreadComputeTime);

		cudaThreadSynchronize();

		
		delete bTrans;
	}*/
}

void NVMatrix::subVector(NVMatrix* vec, int numBlocks, int numThreadsPerBlock){
	addVector(vec, -1, this, numBlocks, numThreadsPerBlock);
}
void NVMatrix::addVector(NVMatrix* vec, int numBlocks, int numThreadsPerBlock){
	addVector(vec, 1, this, numBlocks, numThreadsPerBlock);
}

void NVMatrix::addVector(NVMatrix* vec, float scaleVec, NVMatrix* target, \
		int numBlocks, int numThreadsPerBlock){
	assert(vec->getNumRows() == 1 || vec->getNumCols() == 1);
	assert(vec->getNumRows() == _numRows || vec->getNumCols() == _numCols);
	const unsigned int width = _numCols;
	const unsigned int height = _numRows;
	kAddColVector<<<numBlocks, numThreadsPerBlock>>>(_devData, vec->getDevData(), \
			target->getDevData(), width, height, scaleVec);
	cudaThreadSynchronize();

}

void NVMatrix::addRowVector(NVMatrix* vec, \
         int numBlocks, int numThreadsPerBlock){
	assert(vec->getNumRows() == 1 || vec->getNumCols() == 1);
	assert(vec->getNumRows() == _numRows || vec->getNumCols() == _numCols);
	const unsigned int width = _numCols;
	const unsigned int height = _numRows;
	kAddRowVector<<<numBlocks, numThreadsPerBlock>>>(_devData, vec->getDevData(), \
			_devData, width, height, 1);
	cudaThreadSynchronize();
	
}

void NVMatrix::subtractFromScalar(float scalar, NVMatrix* target, int numBlocks, \
		int numThreadsPerBlock) { 
	kSubtractFromScalar<<<numBlocks, numThreadsPerBlock>>>(_devData, \
			target->getDevData(),_numElements);
	cudaThreadSynchronize();
}

void NVMatrix::subtractFromScalar(float scalar, int numBlocks, \
		int numThreadsPerBlock) {
	subtractFromScalar(scalar, this, numBlocks, numThreadsPerBlock);
}



void NVMatrix::apply(NVMatrix::FUNCTIONS f, NVMatrix *target, int numBlocks, \
		int numThreadsPerBlock){
	dim3 blocks(numBlocks, 1, 1);
	dim3 threads(numThreadsPerBlock, 1, 1);

	if(f == NVMatrix::SOFTMAX){
		kSoftmax<<<blocks, threads>>>(_devData, target->getDevData(), _numCols);
	}else if(f == NVMatrix::RECIPROCAL) {
		kReciprocal<<<blocks, threads>>>(_devData, target->getDevData(), \
				_numElements);
	}else if(f == NVMatrix::LOG) {
		kLog<<<blocks, threads>>>(_devData, target->getDevData(), \
				_numCols);
	}
	cudaThreadSynchronize();
}

void NVMatrix::apply(NVMatrix::FUNCTIONS f, int numBlocks, int numThreadsPerBlock) {
	apply(f, this, numBlocks, numThreadsPerBlock);
}


NVMatrix* NVMatrix::sumCol(int numBlocks, int numThreadsPerBlock){
	NVMatrix *sumVec = new NVMatrix( _numRows, 1);
	sumCol(sumVec, numBlocks, numThreadsPerBlock);
	return sumVec;
}

void NVMatrix::sumCol(NVMatrix* target, int numBlocks, int numThreadsPerBlock){
	unsigned int width = _numCols;
	const int height = _numRows;
	kDumbSumCols<<<numBlocks, numThreadsPerBlock>>>(_devData, target->getDevData(), \
			width, height);
	cudaThreadSynchronize();
}

void NVMatrix::sumRow(NVMatrix* target, int numBlocks, int numThreadsPerBlock){
	unsigned int width = _numCols;
	const int height = _numRows;
	kDumbSumRows<<<numBlocks, numThreadsPerBlock>>>(_devData, \
			target->getDevData(), width, height);
	cudaThreadSynchronize();
}

void NVMatrix::sumRowWithInterval(NVMatrix* target, int interval,\
		int numBlocks, int numThreadsPerBlock){
	unsigned int width = _numRows;
	const int height = _numCols;
	kSumRowInterval<<<numBlocks, numThreadsPerBlock>>>(_devData, \
			target->getDevData(), width, height, interval);
	cudaThreadSynchronize();
}

//return each row max value not pos
NVMatrix* NVMatrix::maxCol(int numBlocks, int numThreadsPerBlock){
	NVMatrix *maxVec = new NVMatrix(_numRows, 1);
	unsigned int width = _numCols;
	const int height = _numRows;
	kDumbMaxCols<<<numBlocks, numThreadsPerBlock>>>(_devData, maxVec->getDevData(), \
			width, height);
	cudaThreadSynchronize();
	return maxVec;
}

void NVMatrix::eltWiseMultByVector(NVMatrix* vec, NVMatrix* target, \
		int numBlocks, int numThreadsPerBlock) {

	assert(vec->getNumCols() == 1);

	const unsigned int width =  _numCols;
	const unsigned int height = _numRows;

	kMultByColVector<<<numBlocks, numThreadsPerBlock>>>(_devData, \
			vec->getDevData(), target->getDevData(), width, height);
	cudaThreadSynchronize();
}

void NVMatrix::eltWiseDivideByVector(NVMatrix* vec, int numBlocks, \
		int numThreadsPerBlock) {
	eltWiseDivideByVector(vec, this, numBlocks, numThreadsPerBlock);
}

void NVMatrix::eltWiseDivideByVector(NVMatrix* vec, NVMatrix* target, \
		int numBlocks, int numThreadsPerBlock) {
	NVMatrix* vecRecip = new NVMatrix(vec->getNumRows(), vec->getNumCols());
	vec->apply(NVMatrix::RECIPROCAL, vecRecip, numBlocks, numThreadsPerBlock);
	eltWiseMultByVector(vecRecip, target, numBlocks, numThreadsPerBlock);

	delete vecRecip;
}

void NVMatrix::eltWiseMult(NVMatrix* b, NVMatrix* target, \
		int numBlocks, int numThreadsPerBlock) {

	assert(b->getNumCols() == _numCols);

	const unsigned int width =  _numCols;
	const unsigned int height = _numRows;

	kMult<<<numBlocks, numThreadsPerBlock>>>(_devData, \
			b->getDevData(), target->getDevData(), width, height);
	cudaThreadSynchronize();
}
void NVMatrix::eltWiseMult(NVMatrix* b, int numBlocks, int numThreadsPerBlock) {
	eltWiseMult(b, this, numBlocks, numThreadsPerBlock);
}

void NVMatrix::addSum(NVMatrix* b, NVMatrix* c, float scaleThis, \
		float scaleB, float scaleC, int numBlocks, int numThreadsPerBlock){
	this->add(b, scaleThis, scaleB, numBlocks, numThreadsPerBlock);	
	this->add(c, 1, scaleC, numBlocks, numThreadsPerBlock);	
}

void NVMatrix::add(NVMatrix* b, float scaleThis, float scaleB, int numBlocks, \
		int numThreadsPerBlock){
	assert(this->isSameDims(b));
	unsigned int width = _numCols;
	unsigned int height = _numRows;
	kAdd<<<numBlocks, numThreadsPerBlock>>>(this->getDevData(), b->getDevData(), \
			this->getDevData(), scaleThis, scaleB, width, height);
	cudaThreadSynchronize();
}

void NVMatrix::showValue(string name){

	float* tmp_yh = new float[_numElements];
	this->copyToHost(tmp_yh, _numElements);
	cout << "-------------"<< name << "--------------" << endl;
	cout << _numRows << ":" << _numCols << endl;
	for(int i = 0; i < _numRows; i++){
		for(int j = 0; j < _numCols; j++){
			cout << tmp_yh[i * _numCols + j] << " ";
			if(j != 0 && j % (_numCols) == _numCols  - 1)
				cout << endl;
			if(_numCols == 1)
				cout << endl;
		}
	}
	delete[] tmp_yh;
}

void NVMatrix::reValue(float value){
	int length = this->getNumRows() * this->getNumCols();
	float* tmp_yh = new float[length];
	for(int i = 0; i < length; i++){
		tmp_yh[i] = value;
	}
	this->copyFromHost(tmp_yh, length);
	delete[] tmp_yh;
}

void NVMatrix::reValue(int value){
	int length = this->getNumRows() * this->getNumCols();
	float* tmp_yh = new float[length];
	for(int i = 0; i < length; i++){
		tmp_yh[i] = i % value;
	}
	this->copyFromHost(tmp_yh, length);
	delete[] tmp_yh;
}




















