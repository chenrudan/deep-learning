/*
 * filename: sigmoid_layer.cu
 */

#include <time.h>
#include "sigmoid_layer.cuh"
#include "layer_kernel.cuh"

using namespace std;

SigmoidLayer::SigmoidLayer(pars* network){

	this->_minibatch_size           = network->minibatch_size;

	cublasCreate(&handle);
}

SigmoidLayer::~SigmoidLayer() {

	delete  _y; 
	delete  _dE_dy;
	delete _dE_dx_sigmoid;
}

void SigmoidLayer::initCuda() {

	this->_y               = new NVMatrix(_minibatch_size, _num_out);

	this->_dE_dy           = new NVMatrix(_y);
}

void SigmoidLayer::computeOutputs(NVMatrix* x){ 
	x->apply(NVMatrix::SIGMOID, _y);
}

void SigmoidLayer::computeDerivsOfInput(NVMatrix* dE_dx){

	_y->subtractFromScalar(1, dE_dx);

	dE_dx->eltWiseMult(_y);

	dE_dx->eltWiseMult(_dE_dy);
}


