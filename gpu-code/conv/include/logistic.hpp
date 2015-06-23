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

private:
	FullConnectParam* _fcp;
	Dtype* h_labels;
	Dtype* y_CPU;
	Dtype* correct_probs;
	Matrix<Dtype>* d_max_pos_of_out;
	Dtype* h_max_pos_of_out;

	
};

#include "../src/logistic.cu"

#endif
