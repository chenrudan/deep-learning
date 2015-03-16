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

NVMatrix::NVMatrix(NVMatrix::ALLOCPALCE a){
	_init(0, 0, a);
}
NVMatrix::NVMatrix(int numRows, int numCols, \
		NVMatrix::ALLOCPALCE a){
	_init(numRows, numCols, a);
}
NVMatrix::NVMatrix(const Matrix* like, bool copy, \
		NVMatrix::ALLOCPALCE a){
	_init(like->getNumRows(), like->getNumCols(), a);
	if (copy) {
		copyFromHost(like);
	}
}
NVMatrix::NVMatrix(const NVMatrix* like, bool copy, \
		NVMatrix::ALLOCPALCE a){
	_init(like->getNumRows(), like->getNumCols(), a);
	if (copy) {
		copyFromDevice(like);
	}
}
// not copy the data
NVMatrix::NVMatrix(const Matrix* like, NVMatrix::ALLOCPALCE a) {
	_init(like->getNumRows(), like->getNumCols(), a);
}
NVMatrix::NVMatrix(const NVMatrix* like, NVMatrix::ALLOCPALCE a) {
	_init(like->getNumRows(), like->getNumCols(), a);
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

void NVMatrix::_init(unsigned int numRows, unsigned int numCols, \
	NVMatrix::ALLOCPALCE a) {
	_numRows = numRows;
	_numCols = numCols;
	_numElements = numRows * numCols;
	_ownsData = true;
	if (_numElements > 0) {
		cudaError_t status;
		if(a == ALLOC_ON_GPU_MEMORY){
			status = cudaMalloc((void**) &_devData, \
				_numRows * _numCols * sizeof(float));
		}
		else if(a == ALLOC_ON_UNIFIED_MEMORY){
			status = cudaMallocManaged(&_devData, \
				_numRows * _numCols * sizeof(float));
		}
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

float* NVMatrix::slice(float* &out, int rowStart){
	out = _devData + rowStart * _numCols;
	return out;
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

//不能直接在自身进行转置，需要传入参数或者返回新建的转置值
void NVMatrix::getTranspose(NVMatrix* target){
	
	const unsigned int width = _numCols;
	const unsigned int height = _numRows;
	const int numBlocksX = DIVUP(width, ADD_BLOCK_SIZE);
	assert(numBlocksX < NUM_BLOCKS_MAX);
	const int numBlocksY = max(1, min(DIVUP(height, ADD_BLOCK_SIZE), \
				NUM_BLOCKS_MAX));
	dim3 gridSize(numBlocksX, numBlocksY, 1); 
	dim3 blockSize(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1); 
	
	kTranspose<<<gridSize, blockSize>>>(_devData, \
				target->getDevData(), width, height);
	cudaThreadSynchronize();
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

void NVMatrix::subColVector(NVMatrix* vec){
	addColVector(vec, -1, this);
}
void NVMatrix::addColVector(NVMatrix* vec){
	addColVector(vec, 1, this);
}

void NVMatrix::addColVector(NVMatrix* vec, float scaleVec, NVMatrix* target){

	NVMatrix* oriTrans = new NVMatrix(_numCols, _numRows);
	this->getTranspose(oriTrans);
	oriTrans->addRowVector(vec);
	oriTrans->getTranspose(target);
	delete oriTrans;

}

void NVMatrix::addRowVector(NVMatrix* vec){
	addRowVector(vec, 1, this);	
}

void NVMatrix::addRowVector(NVMatrix* vec, float scaleVec, NVMatrix* target){
	assert(vec->getNumRows() == 1 || vec->getNumCols() == 1);
	assert(vec->getNumRows() == _numRows || vec->getNumCols() == _numCols);
	const unsigned int width = _numCols;
	const unsigned int height = _numRows;

	//表达成了矩阵的结构，就分开处理算了,block和thread的x维控制列数
	const int numBlocksX = DIVUP(width, ADD_BLOCK_SIZE);
	assert(numBlocksX < NUM_BLOCKS_MAX);
	const int numBlocksY = max(1, min(DIVUP(height, ADD_BLOCK_SIZE), \
				NUM_BLOCKS_MAX));
	dim3 gridSize(numBlocksX, numBlocksY, 1); 
	dim3 blockSize(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1); 

	//cudaMemcpyToSymbol("WIDTH", &width, sizeof(width));
	//cudaMemcpyToSymbol("HEIGHT", &height, sizeof(height));
	//cudaMemcpyToSymbol("SCALE_VEC", &scaleVec, sizeof(scaleVec));

	kAddRowVector<<<gridSize, blockSize>>>(_devData, vec->getDevData(), \
			target->getDevData(), width, height, scaleVec);
	cudaThreadSynchronize();
	
}

void NVMatrix::subtractFromScalar(float scalar, NVMatrix* target) { 

	const unsigned int width = _numCols;
	const unsigned int height = _numRows;
	const int numBlocksX = DIVUP(width, ADD_BLOCK_SIZE);
	assert(numBlocksX < NUM_BLOCKS_MAX);
	const int numBlocksY = max(1, min(DIVUP(height, ADD_BLOCK_SIZE), \
				NUM_BLOCKS_MAX));
	dim3 gridSize(numBlocksX, numBlocksY, 1); 
	dim3 blockSize(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1); 
	
	kSubtractFromScalar<<<gridSize, blockSize>>>(_devData, scalar, \
			target->getDevData(), width, height);
	cudaThreadSynchronize();
}

void NVMatrix::subtractFromScalar(float scalar) {
	subtractFromScalar(scalar, this);
}



void NVMatrix::apply(NVMatrix::FUNCTIONS f, NVMatrix *target){
	
	const unsigned int width = _numCols;
	const unsigned int height = _numRows;
	const int numBlocksX = DIVUP(width, ADD_BLOCK_SIZE);
	assert(numBlocksX < NUM_BLOCKS_MAX);
	const int numBlocksY = max(1, min(DIVUP(height, ADD_BLOCK_SIZE), \
				NUM_BLOCKS_MAX));
	dim3 gridSize(numBlocksX, numBlocksY, 1); 
	dim3 blockSize(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1); 

	if(f == NVMatrix::SOFTMAX){
		//一个block只计算一行数据
		gridSize = dim3(1, height, 1);
		blockSize = dim3(numBlocksX * ADD_BLOCK_SIZE, 1, 1);
		kSoftmax<<<gridSize, blockSize, sizeof(float) * width>>>(_devData, \
				_numCols, _numRows);
	}else if(f == NVMatrix::RECIPROCAL) {
		kReciprocal<<<gridSize, blockSize>>>(_devData, target->getDevData(), \
				width, height);
	}else if(f == NVMatrix::LOG) {
		kLog<<<gridSize, blockSize>>>(_devData, target->getDevData(), \
				width, height);
	}
	cudaThreadSynchronize();
}

void NVMatrix::apply(NVMatrix::FUNCTIONS f) {
	apply(f, this);
}


NVMatrix* NVMatrix::sumCol(){
	NVMatrix *sumVec = new NVMatrix( _numRows, 1);
	sumCol(sumVec);
	return sumVec;
}

void NVMatrix::sumCol(NVMatrix* target){
	const unsigned int width = _numCols;
	const unsigned int height = _numRows;
	const int numBlocksX = DIVUP(width, ADD_BLOCK_SIZE);
	assert(numBlocksX < NUM_BLOCKS_MAX);
	dim3 gridSize(1, height, 1); 
	dim3 blockSize(numBlocksX * ADD_BLOCK_SIZE, 1, 1); 
	
	kDumbSumCols<<<gridSize, blockSize, sizeof(float) * width>>>(_devData, \
			target->getDevData(), width, height);
	cudaThreadSynchronize();
}

void NVMatrix::sumRow(NVMatrix* target){
	NVMatrix* trans = new NVMatrix(_numCols, _numRows);
	this->getTranspose(trans);
	trans->sumCol(target);
	delete trans;
}

//位置下标从0开始
void NVMatrix::maxPosInRow(NVMatrix* maxVec){
	const unsigned int width = _numCols;
	const unsigned int height = _numRows;
	const int numBlocksX = DIVUP(width, ADD_BLOCK_SIZE);
	assert(numBlocksX < NUM_BLOCKS_MAX);
	dim3 gridSize(1, height, 1); 
	dim3 blockSize(numBlocksX * ADD_BLOCK_SIZE, 1, 1); 
	
	kDumbMaxPosInRow<<<gridSize, blockSize, sizeof(float) * width>>>(_devData, \
			maxVec->getDevData(), width, height);
	cudaThreadSynchronize();
}

void NVMatrix::eltWiseMultByColVector(NVMatrix* vec){
	eltWiseMultByColVector(vec, this);
}

void NVMatrix::eltWiseMultByColVector(NVMatrix* vec, NVMatrix* target) {

	assert(vec->getNumCols() == 1);

	const unsigned int width = _numCols;
	const unsigned int height = _numRows;
	const int numBlocksX = DIVUP(width, ADD_BLOCK_SIZE);
	assert(numBlocksX < NUM_BLOCKS_MAX);
	const int numBlocksY = max(1, min(DIVUP(height, ADD_BLOCK_SIZE), \
				NUM_BLOCKS_MAX));
	dim3 gridSize(numBlocksX, numBlocksY, 1); 
	dim3 blockSize(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1); 

	kMultByColVector<<<gridSize, blockSize>>>(_devData, \
			vec->getDevData(), target->getDevData(), width, height);
	cudaThreadSynchronize();
}

void NVMatrix::eltWiseDivideByColVector(NVMatrix* vec) {
	eltWiseDivideByColVector(vec, this);
}

void NVMatrix::eltWiseDivideByColVector(NVMatrix* vec, NVMatrix* target) {
	NVMatrix* vecRecip = new NVMatrix(vec->getNumRows(), vec->getNumCols());
	vec->apply(NVMatrix::RECIPROCAL, vecRecip);
	eltWiseMultByColVector(vecRecip, target);

	delete vecRecip;
}

void NVMatrix::eltWiseMult(NVMatrix* b, NVMatrix* target) {

	assert(b->getNumCols() == _numCols);

	const unsigned int width = _numCols;
	const unsigned int height = _numRows;
	const int numBlocksX = DIVUP(width, ADD_BLOCK_SIZE);
	assert(numBlocksX < NUM_BLOCKS_MAX);
	const int numBlocksY = max(1, min(DIVUP(height, ADD_BLOCK_SIZE), \
				NUM_BLOCKS_MAX));
	dim3 gridSize(numBlocksX, numBlocksY, 1); 
	dim3 blockSize(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1); 

	kMult<<<gridSize, blockSize>>>(_devData, \
			b->getDevData(), target->getDevData(), width, height);
	cudaThreadSynchronize();
}

void NVMatrix::eltWiseMult(NVMatrix* b) {
	eltWiseMult(b, this);
}

void NVMatrix::addSum(NVMatrix* b, NVMatrix* c, float scaleThis, \
		float scaleB, float scaleC){
	this->add(b, scaleThis, scaleB);	
	this->add(c, 1, scaleC);	
}

void NVMatrix::add(NVMatrix* b, float scaleThis, float scaleB){
	assert(this->isSameDims(b));
	const unsigned int width = _numCols;
	const unsigned int height = _numRows;
	const int numBlocksX = DIVUP(width, ADD_BLOCK_SIZE);
	assert(numBlocksX < NUM_BLOCKS_MAX);
	const int numBlocksY = max(1, min(DIVUP(height, ADD_BLOCK_SIZE), \
				NUM_BLOCKS_MAX));
	dim3 gridSize(numBlocksX, numBlocksY, 1); 
	dim3 blockSize(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1); 
	
	kAdd<<<gridSize, blockSize>>>(this->getDevData(), b->getDevData(), \
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




















