///
/// \file relu_layer.cuh
/// @brief 实现了对输入每一个点求relu

#ifndef RELU_LAYER_H_
#define RELU_LAYER_H_

#include <iostream>
#include "layer.hpp"

template <typename Dtype>
class ReluLayer : public Layer<Dtype> {

public:
	
	ReluLayer(Param* fcp);
	~ReluLayer();

	void initCuda();
	void computeOutput(Matrix<Dtype>* x);
	void computeDerivsOfInput(Matrix<Dtype>* dE_dx);

private:
	Param* _p;
	Matrix<int> *_record;
};


#include "../src/relu_layer.cu"




































#endif
