/*
 * filename: relu_layer.cu
 */

#include <time.h>
#include "relu_layer.cuh"
#include "layer_kernel.cuh"

using namespace std;

reluLayer::ReluLayer(pars* network){

	this->_minibatch_size           = network->minibatch_size;

	cublasCreate(&handle);
}

ReluLayer::~ReluLayer() {

	delete  _y; 
	delete  _dE_dy;
}

void ReluLayer::initCuda() {

	this->_y               = new NVMatrix(_minibatch_size, _num_out);

	this->_dE_dy           = new NVMatrix(_y);
}

void ReluLayer::computeOutputs(NVMatrix* x){ 
	x->apply(NVMatrix::relu, _y);
}

void ReluLayer::computeDerivsOfInput(NVMatrix* dE_dx){

	_y->subtractFromScalar(1, dE_dx);

	dE_dx->eltWiseMult(_y);

	dE_dx->eltWiseMult(_dE_dy);
}


