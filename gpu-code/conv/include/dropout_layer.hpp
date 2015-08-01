///
/// \file dropout_layer.cuh
/// @brief 实现了对输入每一个点求dropout

#ifndef DROPOUT_LAYER_H_
#define DROPOUT_LAYER_H_

#include <iostream>
#include <curand.h>
#include "layer.hpp"

template <typename Dtype>
class DropoutLayer : public Layer<Dtype> {

public:
	
	DropoutLayer(Param* fcp);
	~DropoutLayer();

	void initCuda();
	void computeOutputs(Matrix<Dtype>* x);
	void computeDerivsOfInput(Matrix<Dtype>* dE_dx);

private:
	Param* _p;
	Matrix<int> *_drop_record;  ///>记录该点是否被丢弃
	Matrix<curandState> *_drop_rand_probs; ///>记录该点被丢弃的概率，与0.5比较
	bool _is_set_up;  ///>随机数初始化
};


#include "../src/dropout_layer.cu"
#endif
