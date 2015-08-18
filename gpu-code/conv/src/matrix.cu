///
/// \file matrix.cu
/// \brief 矩阵类源文件

#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "matrix.hpp"
#include "matrix_kernel.hpp"

using namespace std;

template <typename Dtype>
Matrix<Dtype>::Matrix(int num_row, int num_col){
	_init(num_row, num_col);
}

template <typename Dtype>
Matrix<Dtype>::Matrix(const Matrix<Dtype>* like, bool copy){
	_init(like->getNumRows(), like->getNumCols());
	if (copy) {
		copyFromDevice(like);
	}
}

template <typename Dtype>
Matrix<Dtype>::Matrix(const Matrix<Dtype>* like) {
	_init(like->getNumRows(), like->getNumCols());
}

template <typename Dtype>
Matrix<Dtype>::~Matrix(){
	if(this->_is_own_data && this->_amount > 0){
		cudaFree(this->_data_value);
	}
}

template <typename Dtype>
void Matrix<Dtype>::_init(int num_row, int num_col) {
	this->_shape.push_back(num_row);
	this->_shape.push_back(num_col);
	this->_amount = num_row * num_col;
	this->_is_own_data = true;
	if (this->_amount > 0) {
		cudaError_t status;
		status = cudaMalloc((void**) &this->_data_value, \
				this->_amount * sizeof(Dtype));
		/*
		else if(a == ALLOC_ON_UNIFIED_MEMORY){
			status = cudaMallocManaged(&this->_data_value, \
				this->_shape[0] * this->_shape[1] * sizeof(Dtype));
		}*/
		if (status != cudaSuccess) {
			fprintf(stderr, "!!!! device memory allocation error\n");
			exit(EXIT_FAILURE);
		}
	} 
}


template <typename Dtype>
void Matrix<Dtype>::getTranspose(Matrix<Dtype>* target){
	
	const int width = this->_shape[1];
	const int height = this->_shape[0];
	const int num_blocks_x = DIVUP(width, ADD_BLOCK_SIZE);
	assert(num_blocks_x < NUM_BLOCKS_MAX);
	const int num_blocks_y = max(1, min(DIVUP(height, ADD_BLOCK_SIZE), \
				NUM_BLOCKS_MAX));
	dim3 grid_size(num_blocks_x, num_blocks_y, 1); 
	dim3 block_size(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1); 
	
	kTranspose<Dtype><<<grid_size, block_size>>>(this->_data_value, \
				target->getDevData(), width, height);
	cudaThreadSynchronize();
	cudaCheckError();
}

template <typename Dtype>
void Matrix<Dtype>::rightMult(Matrix<Dtype>* b, float scale_AB, \
		Matrix<Dtype> *target, cublasHandle_t& handle) {

	clock_t t = clock();

	int m = this->_shape[0];
	int k = this->_shape[1];
	int n = b->getNumCols();
	float scale_tar = 0;
	assert(k == b->getNumRows());
	//列主
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &scale_AB, \
				b->getDevData(), n, this->_data_value, k, \
				&scale_tar, target->getDevData(), n);
}

template <typename Dtype>
void Matrix<Dtype>::addColVector(Matrix<Dtype>* vec){
	addColVector(vec, 1, this);
}

template <typename Dtype>
void Matrix<Dtype>::addColVector(Matrix<Dtype>* vec, float scaleVec, Matrix<Dtype>* target){

	Matrix<Dtype>* ori_trans = new Matrix(this->_shape[1], this->_shape[0]);
	this->getTranspose(ori_trans);
	ori_trans->addRowVector(vec);
	ori_trans->getTranspose(target);
	delete ori_trans;
}

template <typename Dtype>
void Matrix<Dtype>::addRowVector(Matrix<Dtype>* vec){
	addRowVector(vec, 1, this);	
}

template <typename Dtype>
void Matrix<Dtype>::addRowVector(Matrix<Dtype>* vec, float scaleVec, Matrix<Dtype>* target){
	assert(vec->getNumRows() == 1 || vec->getNumCols() == 1);
	assert(vec->getNumRows() == this->_shape[0] || vec->getNumCols() == this->_shape[1]);
	const int width = this->_shape[1];
	const int height = this->_shape[0];

	//表达成了矩阵的结构，就分开处理算了,block和thread的x维控制列数
	const int num_blocks_x = DIVUP(width, ADD_BLOCK_SIZE);
	assert(num_blocks_x < NUM_BLOCKS_MAX);
	const int num_blocks_y = max(1, min(DIVUP(height, ADD_BLOCK_SIZE), \
				NUM_BLOCKS_MAX));
	dim3 grid_size(num_blocks_x, num_blocks_y, 1); 
	dim3 block_size(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1); 

	kAddRowVector<Dtype><<<grid_size, block_size>>>(this->_data_value, vec->getDevData(), \
			target->getDevData(), width, height, scaleVec);
	cudaThreadSynchronize();
	cudaCheckError();
	
}

template <typename Dtype>
void Matrix<Dtype>::subtractFromScalar(float scalar, Matrix<Dtype>* target) { 

	const int width = this->_shape[1];
	const int height = this->_shape[0];
	const int num_blocks_x = DIVUP(width, ADD_BLOCK_SIZE);
	assert(num_blocks_x < NUM_BLOCKS_MAX);
	const int num_blocks_y = max(1, min(DIVUP(height, ADD_BLOCK_SIZE), \
				NUM_BLOCKS_MAX));
	dim3 grid_size(num_blocks_x, num_blocks_y, 1); 
	dim3 block_size(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1); 
	
	kSubtractFromScalar<Dtype><<<grid_size, block_size>>>(this->_data_value, scalar, \
			target->getDevData(), width, height);
	cudaThreadSynchronize();
	cudaCheckError();
}

template <typename Dtype>
void Matrix<Dtype>::subtractFromScalar(float scalar) {
	subtractFromScalar(scalar, this);
}

template <typename Dtype>
void Matrix<Dtype>::apply(Matrix<Dtype>::FUNCTIONS f, Matrix<Dtype> *target){
	
	const int width = this->_shape[1];
	const int height = this->_shape[0];
	const int num_blocks_x = DIVUP(width, ADD_BLOCK_SIZE);
	assert(num_blocks_x < NUM_BLOCKS_MAX);
	const int num_blocks_y = max(1, min(DIVUP(height, ADD_BLOCK_SIZE), \
				NUM_BLOCKS_MAX));
	dim3 grid_size(num_blocks_x, num_blocks_y, 1); 
	dim3 block_size(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1); 

	if(f == Matrix<Dtype>::SOFTMAX){
		//一个block只计算一行数据
		grid_size = dim3(1, height, 1);
		block_size = dim3(num_blocks_x * ADD_BLOCK_SIZE, 1, 1);
		kSoftmax<Dtype><<<grid_size, block_size, sizeof(Dtype) * width>>>(this->_data_value, \
				target->getDevData(), this->_shape[1], this->_shape[0]);
	}else if(f == Matrix<Dtype>::RECIPROCAL) {
		kReciprocal<Dtype><<<grid_size, block_size>>>(this->_data_value, target->getDevData(), \
				width, height);
	}else if(f == Matrix<Dtype>::LOG) {
		kLog<Dtype><<<grid_size, block_size>>>(this->_data_value, target->getDevData(), \
				width, height);
	}else if(f == Matrix<Dtype>::SIGMOID) {
		kSigmoid<Dtype><<<grid_size, block_size>>>(this->_data_value, target->getDevData(), \
				width, height);
	}
	cudaThreadSynchronize();
	cudaCheckError();
}

template <typename Dtype>
void Matrix<Dtype>::applyRelu(Matrix<Dtype> *target, Matrix<int>* record, \
		bool direction){
	const int width = this->_shape[1];
	const int height = this->_shape[0];
	const int num_blocks_x = DIVUP(width, ADD_BLOCK_SIZE);
	assert(num_blocks_x < NUM_BLOCKS_MAX);
	const int num_blocks_y = max(1, min(DIVUP(height, ADD_BLOCK_SIZE), \
				NUM_BLOCKS_MAX));
	dim3 grid_size(num_blocks_x, num_blocks_y, 1); 
	dim3 block_size(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1);

	if(direction)
		kRelu<Dtype><<<grid_size, block_size>>>(this->_data_value, \
				target->getDevData(), record->getDevData(), width, height);	
	else
		kReluBack<Dtype><<<grid_size, block_size>>>(this->_data_value, \
				target->getDevData(), record->getDevData(), width, height);	
	cudaThreadSynchronize();
	cudaCheckError();
}

template <typename Dtype>
void Matrix<Dtype>::applyDropout(Matrix<Dtype> *target, Matrix<int>* record, \
		Matrix<curandState>* rand_probs, bool is_set_up){

	const int width = this->_shape[1];
	const int height = this->_shape[0];
	const int num_blocks_x = DIVUP(width, ADD_BLOCK_SIZE);
	assert(num_blocks_x < NUM_BLOCKS_MAX);
	const int num_blocks_y = max(1, min(DIVUP(height, ADD_BLOCK_SIZE), \
				NUM_BLOCKS_MAX));
	dim3 grid_size(num_blocks_x, num_blocks_y, 1); 
	dim3 block_size(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1);

	if(is_set_up == false){
		kSetUpCurand<Dtype><<<grid_size, block_size>>>(rand_probs->getDevData(), \
				width, height);	
		cudaThreadSynchronize();
		cudaCheckError();
	
	}

	kDropout<Dtype><<<grid_size, block_size>>>(this->_data_value, \
			target->getDevData(), record->getDevData(), \
			rand_probs->getDevData(), width, height);	
	cudaThreadSynchronize();
	cudaCheckError();
}

template <typename Dtype>
void Matrix<Dtype>::apply(Matrix<Dtype>::FUNCTIONS f) {
	apply(f, this);
}

template <typename Dtype>
void Matrix<Dtype>::sumCol(Matrix<Dtype>* target){
	const int width = this->_shape[1];
	const int height = this->_shape[0];

	kDumbSumCols<Dtype><<<height, 1024, sizeof(Dtype) * width>>>(this->_data_value, \
			target->getDevData(), width, height);
	cudaThreadSynchronize();
	cudaCheckError();
}

template <typename Dtype>
void Matrix<Dtype>::sumRow(Matrix<Dtype>* target){
	Matrix<Dtype>* trans = new Matrix(this->_shape[1], this->_shape[0]);
	this->getTranspose(trans);
	trans->sumCol(target);
	delete trans;
}

//位置下标从0开始
template <typename Dtype>
void Matrix<Dtype>::maxPosInRow(Matrix<Dtype>* maxVec){
	const int width = this->_shape[1];
	const int height = this->_shape[0];
	const int num_blocks_x = DIVUP(width, ADD_BLOCK_SIZE);
	assert(num_blocks_x < NUM_BLOCKS_MAX);
	dim3 grid_size(1, height, 1); 
	dim3 block_size(num_blocks_x * ADD_BLOCK_SIZE, 1, 1); 

	kDumbMaxPosInRow<Dtype><<<grid_size, block_size, \
			sizeof(Dtype) * width>>>(this->_data_value, \
			maxVec->getDevData(), width, height);
	cudaThreadSynchronize();
	cudaCheckError();
}

template <typename Dtype>
void Matrix<Dtype>::eltWiseMult(Matrix<Dtype>* b, Matrix<Dtype>* target) {

	assert(b->getNumCols() == this->_shape[1]);

	const int width = this->_shape[1];
	const int height = this->_shape[0];
	const int num_blocks_x = DIVUP(width, ADD_BLOCK_SIZE);
	assert(num_blocks_x < NUM_BLOCKS_MAX);
	const int num_blocks_y = max(1, min(DIVUP(height, ADD_BLOCK_SIZE), \
				NUM_BLOCKS_MAX));
	dim3 grid_size(num_blocks_x, num_blocks_y, 1); 
	dim3 block_size(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1); 

	kMult<Dtype><<<grid_size, block_size>>>(this->_data_value, \
			b->getDevData(), target->getDevData(), width, height);
	cudaThreadSynchronize();
	cudaCheckError();
}

template <typename Dtype>
void Matrix<Dtype>::eltWiseMult(Matrix<Dtype>* b) {
	eltWiseMult(b, this);
}

template <typename Dtype>
void Matrix<Dtype>::addSum(Matrix<Dtype>* b, Matrix<Dtype>* c, float scaleThis, \
		float scaleB, float scaleC){
	this->add(b, scaleThis, scaleB);	
	this->add(c, 1, scaleC);	
}

template <typename Dtype>
void Matrix<Dtype>::add(Matrix<Dtype>* b, float scale_this, float scale_B){
	assert(this->isSameDims(b));
	const int width = this->_shape[1];
	const int height = this->_shape[0];
	const int num_blocks_x = DIVUP(width, ADD_BLOCK_SIZE);
	assert(num_blocks_x < NUM_BLOCKS_MAX);
	const int num_blocks_y = max(1, min(DIVUP(height, ADD_BLOCK_SIZE), \
				NUM_BLOCKS_MAX));
	dim3 grid_size(num_blocks_x, num_blocks_y, 1); 
	dim3 block_size(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1); 
	
	kAdd<Dtype><<<grid_size, block_size>>>(this->getDevData(), b->getDevData(), \
			this->getDevData(), scale_this, scale_B, width, height);
	cudaThreadSynchronize();
	cudaCheckError();
}


template <typename Dtype>
void Matrix<Dtype>::showValue(string name){

	Dtype* tmp_yh = new Dtype[this->_amount];
	this->copyToHost(tmp_yh, this->_amount);
	cout << "-------------"<< name << "--------------" << endl;
	cout << this->_shape[0] << ":" << this->_shape[1] << endl;
	for(int i = 0; i < this->_shape[0]; i++){
		for(int j = 0; j < this->_shape[1]; j++){
			cout << tmp_yh[i * this->_shape[1] + j] << "\t";
			if(j != 0 && j % (this->_shape[1]) == this->_shape[1]  - 1)
				cout << endl;
			if(this->_shape[1] == 1)
				cout << endl;
		}
	}
	delete[] tmp_yh;
}

template <typename Dtype>
void Matrix<Dtype>::reValue(float value){
	int length = this->getNumRows() * this->getNumCols();
	Dtype* tmp_yh = new Dtype[length];
	for(int i = 0; i < length; i++){
		tmp_yh[i] = value;
	}
	this->copyFromHost(tmp_yh, length);
	delete[] tmp_yh;
}

template <typename Dtype>
void Matrix<Dtype>::reValue(int value){
	int length = this->getNumRows() * this->getNumCols();
	Dtype* tmp_yh = new Dtype[length];
	for(int i = 0; i < length; i++){
		tmp_yh[i] = i % value;
	}
	this->copyFromHost(tmp_yh, length);
	delete[] tmp_yh;
}

template <typename Dtype>
Dtype Matrix<Dtype>::computeNorm(int len){
	Dtype norm_cpu;
	Matrix<Dtype>* norm_gpu = new Matrix<Dtype>(1, 1);
	kComputeNorm<<<1, 1024, sizeof(Dtype)*len>>>(this->_data_value, \
			norm_gpu->getDevData(), len);
	cudaThreadSynchronize();
	cudaCheckError();
	norm_gpu->copyToHost(&norm_cpu, 1);
	delete norm_gpu;
	return norm_cpu;
}

template <typename Dtype>
void Matrix<Dtype>::cropMatToNew(Matrix<Dtype> *tar, const int row_start, \
		const int cropped_height, const int col_start, const int cropped_width){
	kCropImg<<<1, 1024>>>(this->_data_value, tar->getDevData(), row_start, \
			cropped_height, col_start, cropped_width, this->_shape[1]);
	cudaThreadSynchronize();
	cudaCheckError();
}

template <typename Dtype>
Dtype Matrix<Dtype>::getPosValue(int pos){
	Dtype tmp;
	cudaMemcpy(&tmp, this->_data_value + pos, sizeof(Dtype), cudaMemcpyDeviceToHost);
	return tmp;
}

template <typename Dtype>
Dtype Matrix<Dtype>::getFirstPosValue(){
	return getPosValue(0);
}

template <typename Dtype>
void Matrix<Dtype>::subedByUnitMat(){

	const int width = this->_shape[1];
	const int height = this->_shape[0];
	const int num_blocks_x = DIVUP(width, ADD_BLOCK_SIZE);
	assert(num_blocks_x < NUM_BLOCKS_MAX);
	const int num_blocks_y = max(1, min(DIVUP(height, ADD_BLOCK_SIZE), \
				NUM_BLOCKS_MAX));
	dim3 grid_size(num_blocks_x, num_blocks_y, 1);
	dim3 block_size(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1);

	kSubedByUnitMat<Dtype><<<grid_size, block_size>>>(this->getDevData(), \
			this->getDevData(), width, height);
	cudaThreadSynchronize();
	cudaCheckError();
}

template <typename Dtype>
void Matrix<Dtype>::setValueAt(const int height_idx, \
		const int width_idx, const Dtype value){
	int pos = height_idx*this->_shape[1] + width_idx;
	cudaMemcpy(this->_data_value + pos, &value, sizeof(Dtype), \
			cudaMemcpyHostToDevice);
}

template <typename Dtype>
void Matrix<Dtype>::subPortion(Matrix<Dtype>* b, const int b_row, \
			const int b_col){

	const int width = b_col;
	const int height = b_row;
	const int num_blocks_x = DIVUP(width, ADD_BLOCK_SIZE);
	assert(num_blocks_x < NUM_BLOCKS_MAX);
	const int num_blocks_y = max(1, min(DIVUP(height, ADD_BLOCK_SIZE), \
				NUM_BLOCKS_MAX));
	dim3 grid_size(num_blocks_x, num_blocks_y, 1);
	dim3 block_size(ADD_BLOCK_SIZE, ADD_BLOCK_SIZE, 1);

	kSubPortion<Dtype><<<grid_size, block_size>>>(this->getDevData(), \
			b->getDevData()+b_col, this->getDevData(), this->_shape[1], \
			this->_shape[0], width, height);
	cudaThreadSynchronize();
	cudaCheckError();
}

template <typename Dtype>
void Matrix<Dtype>::readPars(string filename){
	ifstream fin1(filename.c_str(), ios::binary);
	int dataLen = this->getNumRows() * this->getNumCols();
	Dtype* tmp = new Dtype[dataLen];
	fin1.read((char*)(tmp), sizeof(Dtype) * dataLen);
	cudaMemcpy(this->getDevData(), tmp, sizeof(Dtype)*dataLen, \
				cudaMemcpyHostToDevice);
	fin1.close();
	delete tmp;
}

template <typename Dtype>
void Matrix<Dtype>::savePars(string filename){
	ofstream fout(filename.c_str(), ios::binary);
	int dataLen = this->getNumRows() * this->getNumCols();
	Dtype* tmp = new Dtype[dataLen];
	cudaMemcpy(tmp, this->getDevData(), sizeof(Dtype)*dataLen, \
				cudaMemcpyDeviceToHost);
	fout.write((char*)(tmp), sizeof(Dtype) * dataLen);
	fout.close();
	delete tmp;
}










