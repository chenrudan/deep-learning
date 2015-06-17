///
/// \file inner_product_layer.cuh
/// @brief 实现了inner product

#ifndef INNER_PRODUCT_LAYER_CUH_
#define INNER_PRODUCT_LAYER_CUH_

#include <iostream>
#include "layer.hpp"
#include "layer_kernel.cuh"

template <typename Dtype>
class InnerProductLayer : public TrainLayer<Dtype> {

public:
	
	InnerProductLayer(InnerParam* fcp);
	~InnerProductLayer();

	void initCuda();
	void computeOutputs(Matrix<Dtype>* x);
	void computeDerivsOfPars(Matrix<Dtype>* x);
	void computeDerivsOfInput(Matrix<Dtype>* dE_dx);

private:
	InnerParam* _fcp;
};

#include "../src/inner_product_layer.cu"





































#endif
