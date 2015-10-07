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

	void saveRecommendW(string filename){
		dE_dw->copyFromHost(dE_dw_CPU, dE_dw->getNumEles());
		dE_dw->savePars(filename);
	}
	void readRecommendW(string filename){
		dE_dw->readPars(filename);
		dE_dw->copyToHost(dE_dw_CPU, dE_dw->getNumEles());
	}
	float* getProbRecord(){
		return y_CPU;
	}
	int* getHLabel(){
		return h_labels;
	}

private:
	FullConnectParam* _fcp;
	Dtype* y_CPU;
	Dtype* x_CPU;
	Dtype* dE_dx_CPU;
	Dtype* w_CPU;
	Dtype* dE_dw_CPU;
	Matrix<Dtype>* dE_dw;
	int *h_labels;
	bool _is_compatible;

};


#include "../src/recommendation_layer.cu"
#endif
