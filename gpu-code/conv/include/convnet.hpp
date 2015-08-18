/// 
/// \file convnet.hpp
/// @brief

#ifndef CONVNET_H_
#define CONVNET_H_

#include <iostream>
#include "layer.hpp"


template <typename Dtype>
class ConvNet : public TrainLayer<Dtype>{

private:


	Matrix<Dtype>* unrolled_x1; ///> 重排x用来计算正向卷积
	Matrix<Dtype>* unranged_y;  
	Matrix<Dtype>* unrolled_x2; ///> 重排x用来求权重导数
	Matrix<Dtype>* ranged_dE_dy;  ///> 重排dE_dy来计算w的导数
	Matrix<Dtype>* ranged_dE_dy2; ///> 重排dE_dy来计算b的导数
	Matrix<Dtype>* unrolled_dE_db_tmp;
	Matrix<Dtype>* dE_db_tmp;
	Matrix<Dtype>* unrolled_conv;
	Matrix<Dtype>* ranged_w;
	Matrix<Dtype>* unranged_in;
	Matrix<Dtype>* padded_x;

	int _filt_pixs;
	int _conv_pixs;
	int _padded_in_pixs;
	
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
