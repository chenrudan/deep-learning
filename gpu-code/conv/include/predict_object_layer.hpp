///
/// \file predict_object_layer.cuh
/// @brief 实现了predict object coord

#ifndef PREDICT_OBJECT_LAYER_CUH_
#define PREDICT_OBJECT_LAYER_CUH_

#include <iostream>
#include "layer.hpp"

template <typename Dtype>
class PredictObjectLayer : public Layer<Dtype>{

public:
	
	PredictObjectLayer(FullConnectParam* fcp);
	~PredictObjectLayer();

	void initCuda();
	double computeError(Matrix<Dtype> *x, Matrix<int>* coord);
	void computeDerivsOfInput(Matrix<Dtype>* dE_dx);

private:
	FullConnectParam* _fcp;
	Dtype *_h_x;
	Dtype *_h_dE_dx;
	int *_h_coord;
};

#include "../src/predict_object_layer.cu"





































#endif
