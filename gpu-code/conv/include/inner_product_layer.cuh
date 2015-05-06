/*
 * filename: inner_product_layer.cuh
 */

#ifndef INNER_PRODUCT_LAYER_H_
#define INNER_PRODUCT_LAYER_H_

#include <iostream>
#include "layer.hpp"
#include "utils.cuh"
#include "matrix.h"
#include "nvmatrix.cuh"

class InnerProductLayer : public Layer {

public:
	
	InnerProductLayer(pars* netWork);
	~InnerProductLayer();

	void initCuda();
	void computeOutputs(NVMatrix* x);
	using Layer::computeDerivsOfPars;
	void computeDerivsOfPars(NVMatrix* x);
	void computeDerivsOfInput(NVMatrix* dE_dx);

private:
	NVMatrix* _dE_dx_sigmoid;

};







































#endif
