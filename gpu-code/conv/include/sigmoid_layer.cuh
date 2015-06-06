/*
 * filename: sigmoid_layer.cuh
 */

#ifndef SIGMOID_LAYER_H_
#define SIGMOID_LAYER_H_

#include <iostream>
#include "layer.hpp"
#include "utils.cuh"
#include "matrix.h"
#include "nvmatrix.cuh"

class SigmoidLayer : public Layer {

public:
	
	SigmoidLayer(pars* netWork);
	~SigmoidLayer();

	void initCuda();
	void computeOutputs(NVMatrix* x);
	void computeDerivsOfInput(NVMatrix* dE_dx);

};







































#endif
