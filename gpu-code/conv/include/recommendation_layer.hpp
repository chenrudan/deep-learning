///
/// \file recommendation_layer.cuh
/// @brief 实现了对输入每一个点求Recommendation

#ifndef Recommendation_LAYER_H_
#define Recommendation_LAYER_H_

#include <iostream>
#include "layer.hpp"

template <typename Dtype>
class RecommendationLayer : public Layer<Dtype> {

public:
	
	RecommendationLayer(FullConnectParam* fcp);
	~RecommendationLayer();

	void initCuda();
	double computeError(Matrix<Dtype>* x, Matrix<int> *labels);
	void computeDerivsOfInput(Matrix<Dtype>* dE_dx);

private:
	FullConnectParam* _fcp;
	Dtype* y_CPU;
	Dtype* x_CPU;
	Dtype* dE_dx_CPU;
	Dtype* w_CPU;
	Dtype* dE_dw_CPU;
	int *h_labels;
};


#include "../src/recommendation_layer.cu"
#endif
