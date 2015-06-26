///
/// \file logistic.cuh
/// @brief 实现了softmax

#ifndef LOGISTIC_CUH_
#define LOGISTIC_CUH_

#include <iostream>
#include "layer.hpp"
#include "layer_kernel.cuh"

template <typename Dtype>
class Logistic : public Layer<Dtype> {

public:
	Logistic(FullConnectParam* fcp);
	~Logistic();
	
	void initCuda();
	void computeOutputs(Matrix<Dtype>* x);
	double computeError(Matrix<Dtype>* labels, int& num_error);
	using Layer<Dtype>::computeDerivsOfInput;
	void computeDerivsOfInput(Matrix<Dtype>* x, Matrix<Dtype>* labels);

	inline Matrix<int>* getResultRecord(){
		_d_record->copyFromHost(_h_record, this->_y->getNumCols() * this->_y->getNumCols());
		return _d_record;
	}
	inline void setRecordToZero(){
		memset(_h_record, 0, sizeof(int) * this->_y->getNumCols() * this->_y->getNumCols());
	}


private:
	FullConnectParam* _fcp;
	Dtype* h_labels;
	Dtype* y_CPU;
	Dtype* correct_probs;
	Matrix<Dtype>* d_max_pos_of_out;
	Dtype* h_max_pos_of_out;

	Matrix<int>* _d_record;  ///>这个变量用来存储最后分类的结果，10*10的矩阵
	int* _h_record;

	
};

#include "../src/logistic.cu"

#endif
