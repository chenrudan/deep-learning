/// 
/// \file convnet.hpp
/// @brief

#ifndef CONVNET_H_
#define CONVNET_H_

#include <iostream>
#include <cudnn.h>
#include "layer.hpp"


template <typename Dtype>
class ConvNet : public TrainLayer<Dtype>{

private:

	Matrix<Dtype>* unfold_dE_db_tmp;
	Matrix<Dtype>* dE_db_tmp;
	Matrix<Dtype>* padded_x;
	Matrix<Dtype>* unfold_x;

	Matrix<Dtype>* unranged_dE_dx;
	Matrix<Dtype>* unranged_dE_dw;
	int _filt_pixs;
	int _conv_pixs;
	int _padded_in_pixs;
	int _in_pixs;
	int _box_in_pixs;
	int _num_box;
	
	ConvParam* _cp;

public:
	ConvNet(ConvParam* cp);
	~ConvNet();

	void initCuda();
	void computeOutput(Matrix<Dtype>* x);
	void computeDerivsOfPars(Matrix<Dtype>* x);
	void computeDerivsOfInput(Matrix<Dtype>* dE_dx);
	
};

#include "../src/convnet.cu"

#endif
