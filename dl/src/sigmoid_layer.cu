///
/// \file sigmoid_layer.cu
/// @brief

#include "sigmoid_layer.hpp"

using namespace std;

template <typename Dtype>
SigmoidLayer<Dtype>::SigmoidLayer(Param* fcp){

	this->_fcp           = fcp;
}

template <typename Dtype>
SigmoidLayer<Dtype>::~SigmoidLayer() {
	delete  this->_y; 
	delete  this->_dE_dy;
}

template <typename Dtype>
void SigmoidLayer<Dtype>::initCuda() {


	ConnectType ct = this->_fcp->getConnectType();
	int col;
	if(ct == PARAM_CONNECT_TYPE_LOCAL)
		col = _fcp->getOutHeight()*_fcp->getOutWidth() \
			   	* this->_fcp->getOutChannel(); 
	else if(ct == PARAM_CONNECT_TYPE_FULL)
		col = this->_fcp->getNumOut(); 
	this->_y             = new Matrix<Dtype>(_fcp->getMinibatchSize(), \
								col);
	this->_dE_dy         = new Matrix<Dtype>(this->_y);
}

template <typename Dtype>
void SigmoidLayer<Dtype>::computeOutput(Matrix<Dtype>* x){ 
	x->apply(Matrix<Dtype>::SIGMOID, this->_y);
}

template <typename Dtype>
void SigmoidLayer<Dtype>::computeDerivsOfInput(Matrix<Dtype>* dE_dx){


	this->_y->subtractFromScalar(1, dE_dx);

	dE_dx->eltWiseMult(this->_y);

	dE_dx->eltWiseMult(this->_dE_dy);

}


