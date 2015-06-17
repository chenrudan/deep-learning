///
/// \file sigmoid_layer.cuh
/// @brief 实现了对输入每一个点求sigmoid

#ifndef SIGMOID_LAYER_H_
#define SIGMOID_LAYER_H_

#include <iostream>
#include "layer.hpp"

template <typename Dtype>
class SigmoidLayer : public Layer<Dtype> {

public:
	
	SigmoidLayer(FullConnectParam* fcp);
	~SigmoidLayer();

	void initCuda();
	void computeOutputs(Matrix<Dtype>* x);
	void computeDerivsOfInput(Matrix<Dtype>* dE_dx);

private:
	FullConnectParam* _fcp;
};


#include "../src/sigmoid_layer.cu"
#endif
