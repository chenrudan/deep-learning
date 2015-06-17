///
/// \file logistic.cuh
/// @brief 实现了softmax

#ifndef LOGISTIC_CUH_
#define LOGISTIC_CUH_

#include <iostream>
#include "layer.hpp"
#include "layer_kernel.cuh"

template <typename Dtype>
class Logistic : public TrainLayer<Dtype> {

public:
	Logistic(InnerParam* fcp);
	~Logistic();
	
	void initCuda();
	void computeOutputs(Matrix<Dtype>* x);
	double computeError(Matrix<Dtype>* labels, int& num_error);
	using TrainLayer<Dtype>::computeDerivsOfPars;
	void computeDerivsOfPars(Matrix<Dtype>* x, Matrix<Dtype>* labels);
	void computeDerivsOfInput(Matrix<Dtype>* dE_dx);

private:
	InnerParam* _fcp;
	

};

#include "../src/logistic.cu"

#endif
