///
/// \file pooling_layer.cuh
/// @brief 实现了pooling

#ifndef POOLING_LAYER_H_
#define POOLING_LAYER_H_

#include <iostream>
#include <cmath>
#include "layer.hpp"
#include "layer_kernel.cuh"

template <typename Dtype>
class PoolingLayer : public Layer<Dtype> {

public:
    PoolingLayer(PoolParam *lcp);

    ~PoolingLayer();

    void initCuda();

    void computeOutputs(Matrix<Dtype>* x);

    void computeDerivsOfInput(Matrix<Dtype>* dE_dx);

private:
    Matrix<int>* _max_pos;
    PoolParam* _lcp;
	Matrix<Dtype>* unranged_dE_dx;
	int _overlap_len;
};

#include "../src/pooling_layer.cu"

#endif

