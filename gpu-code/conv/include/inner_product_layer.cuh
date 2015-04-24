/*
 * filename: inner_product_layer.cuh
 */

#ifndef INNER_PRODUCT_LAYER_H_
#define INNER_PRODUCT_LAYER_H_

#include <iostream>

#include "utils.cuh"
#include "matrix.h"
#include "nvmatrix.cuh"

class InnerProductLayer {

private:
	
	NVMatrix* _w;
	NVMatrix* _bias;
	
	NVMatrix* _dE_dw;
	NVMatrix* _dE_db;
	
	NVMatrix* _y;
	
	float w_lr;
	float b_lr; 
	
	int numIn, numOut;
	
	cublasHandle_t handle;

public:
	
	InnerProductLayer(pars* netWork);
	~InnerProductLayer();

	
	
};







































#endif
