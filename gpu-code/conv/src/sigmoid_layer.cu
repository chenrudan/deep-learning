///
/// \file sigmoid_layer.cu
/// @brief

#include "sigmoid_layer.cuh"

using namespace std;

template <typename Dtype>
SigmoidLayer<Dtype>::SigmoidLayer(FullConnectParam* fcp){

	this->_fcp           = fcp;
}

template <typename Dtype>
SigmoidLayer<Dtype>::~SigmoidLayer() {
	delete  this->_y; 
	delete  this->_dE_dy;
}

template <typename Dtype>
void SigmoidLayer<Dtype>::initCuda() {

	this->_y             = new Matrix<Dtype>(_fcp->getMinibatchSize(), \
								_fcp->getNumOut());

	this->_dE_dy         = new Matrix<Dtype>(this->_y);
}

template <typename Dtype>
void SigmoidLayer<Dtype>::computeOutputs(Matrix<Dtype>* x){ 
	x->apply(Matrix<Dtype>::SIGMOID, this->_y);
}

template <typename Dtype>
void SigmoidLayer<Dtype>::computeDerivsOfInput(Matrix<Dtype>* dE_dx){

	this->_y->subtractFromScalar(1, dE_dx);

	dE_dx->eltWiseMult(this->_y);

	dE_dx->eltWiseMult(this->_dE_dy);
}


