/*
 * filename: layer.hpp
 */
#ifndef LAYER_H_
#define LAYER_H_

#include "utils.cuh"
#include "nvmatrix.cuh"

class Layer {

protected:
	
	//output
	NVMatrix* _y;
	NVMatrix* _dE_dy;
	//parameters
	NVMatrix* _w;
	NVMatrix* _bias;
	NVMatrix* _w_inc;
	NVMatrix* _bias_inc;
	NVMatrix* _dE_dw;
	NVMatrix* _dE_db;
	
	//learning rate
	float _w_lr;
	float _b_lr;
	float _lr_down_scale;

	//mom
	float _momentum; 	
	float _weight_decay;

	int _minibatch_size;
	//if the layer is inner product, than choose this group
	int _num_in;
	int _num_out;
	//if the layer is convolution or pooling, choose this group
	int _in_size;
	int _in_channel;
	int _filter_size;
	int _filter_channel;
	int _out_size;
	int _stride;
	int _pool_size;

	cublasHandle_t handle;

public:
	Layer() {}
	virtual ~Layer() {}	

	virtual void initCuda() {}
	virtual void computeOutput(NVMatrix* x) {}
	virtual void computeDerivsOfPars(NVMatrix* x, NVMatrix* labels = NULL) {}
	virtual void computeDerivsOfInput(NVMatrix* dE_dx) {}

	void updatePars(bool isShow = false) {
		if(isShow == true){
			_w->showValue("w");
			_dE_dw->showValue("dEdw");
			_w_inc->showValue("winc");
			_dE_db->showValue("dEdb");
		}
		_w_inc->addSum(_w, _dE_dw, _momentum, -_weight_decay, \
			            -_w_lr / _minibatch_size);
		_w->add(_w_inc, 1, 1); 

		_bias_inc->add(_dE_db, _momentum, -_b_lr / _minibatch_size);
		_bias->add(_bias_inc, 1, 1);
	}

	inline void transfarLowerPars() {
		_w_lr *= _lr_down_scale;
		_b_lr *= _lr_down_scale;
	}

	inline NVMatrix* getW() {
		return _w;	
	}
	inline NVMatrix* getBias() {
		return _bias;
	}

	inline NVMatrix* getY() {
		return _y; 
	}   
	inline NVMatrix* getDEDY() {
		return _dE_dy;
	}

};



#endif
