///
/// \file layer.hpp
///
#ifndef LAYER_HPP_
#define LAYER_HPP_

#include <cuda_runtime.h>
#include "utils.cuh"
#include "param.h"
#include "matrix.hpp"

template <typename Dtype>
class Layer {

public:
	Layer() {}
	virtual ~Layer() {}

	virtual void initCuda() {}
	virtual void computeOutput(Matrix<Dtype>* x) {}

	virtual void computeDerivsOfInput(Matrix<Dtype>* dE_dx) {}

	inline Matrix<Dtype>* getY() {
		return _y;
	}   
	inline Matrix<Dtype>* getDEDY() {
		return _dE_dy;
	}

protected:
	cublasHandle_t handle;
	Matrix<Dtype>* _y;    ///>每一层的输出
	Matrix<Dtype>* _dE_dy;   ///>每层输出的导数
};

template <typename Dtype>
class TrainLayer : public Layer<Dtype> {

public:
	TrainLayer(TrainParam* tp){
		_tp = tp;
	}
	TrainLayer();
	virtual ~TrainLayer() {}

	virtual void computeDerivsOfPars(Matrix<Dtype>* x) {}

	void updatePars(bool isShow = false) {
		if(isShow == true){
			_w->showValue("w");
			_dE_dw->showValue("dEdw");
			_w_inc->showValue("winc");
			_dE_db->showValue("dEdb");
		}
		_w_inc->addSum(_w, _dE_dw, _tp->getMomentum(), -_tp->getWeightDecay(), \
			            -_tp->getWLR() / _tp->getMinibatchSize());
		_w->add(_w_inc, 1, 1);

		_bias_inc->add(_dE_db, _tp->getMomentum(), \
				-_tp->getBiasLR() / _tp->getMinibatchSize());
		_bias->add(_bias_inc, 1, 1);
	}
	inline Matrix<Dtype>* getW() {
		return _w;
	}
	inline Matrix<Dtype>* getBias() {
		return _bias;
	}

protected:
	Matrix<Dtype>* _w;
	Matrix<Dtype>* _bias;
	Matrix<Dtype>* _w_inc;
	Matrix<Dtype>* _bias_inc;
	Matrix<Dtype>* _dE_dw;
	Matrix<Dtype>* _dE_db;

	TrainParam* _tp;
};



#endif
