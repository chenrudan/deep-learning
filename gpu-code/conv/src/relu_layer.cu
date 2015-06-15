///
/// \file relu_layer.cu
/// @brief

#include <time.h>
#include "relu_layer.cuh"
#include "layer_kernel.cuh"

using namespace std;

template <typename Dtype>
ReluLayer<Dtype>::ReluLayer(FullConnectParam* fcp){

	this->_fcp           = fcp;
}

template <typename Dtype>
ReluLayer<Dtype>::~ReluLayer() {

	delete  this->_y; 
	delete  this->_dE_dy;
}

template <typename Dtype>
void ReluLayer<Dtype>::initCuda() {

	this->_y               = new Matrix<Dtype>(_fcp->getMinibatchSize(), \
								_fcp->getNumOut());

	this->_dE_dy           = new Matrix<Dtype>(this->_y);
}

template <typename Dtype>
void ReluLayer<Dtype>::computeOutputs(Matrix<Dtype>* x){ 
//	x->apply(Matrix<Dtype>::relu, this->_y);
}

template <typename Dtype>
void ReluLayer<Dtype>::computeDerivsOfInput(Matrix<Dtype>* dE_dx){

//	_y->subtractFromScalar(1, dE_dx);

//	dE_dx->eltWiseMult(_y);

//	dE_dx->eltWiseMult(_dE_dy);
}


