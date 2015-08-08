///
/// \file inner_product_layer.cu
/// @brief

#include "inner_product_layer.hpp"

using namespace std;

template <typename Dtype>
InnerProductLayer<Dtype>::InnerProductLayer<Dtype>(InnerParam* fcp) : \
 	TrainLayer<Dtype>((TrainParam*)fcp){
	this->_fcp = fcp;
	this->_layer_type			= INNERPRODUCT;
	cublasCreate(&this->handle);
}

template <typename Dtype>
InnerProductLayer<Dtype>::~InnerProductLayer<Dtype>() {

	delete this->_w; 
	delete this->_w_inc;
	delete this->_bias;
	delete this->_bias_inc;

	delete this->_y; 
	delete this->_dE_dy;
	delete this->_dE_db;
	delete this->_dE_dw;
	
	cublasDestroy(this->handle);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::initCuda() {

	this->_w            = new Matrix<Dtype>(this->_fcp->getNumIn(), this->_fcp->getNumOut());
	this->_bias         = new Matrix<Dtype>(1, this->_fcp->getNumOut());

	this->_y            = new Matrix<Dtype>(this->_fcp->getMinibatchSize(), this->_fcp->getNumOut());
	
	this->_dE_dy        = new Matrix<Dtype>(this->_y);
	this->_dE_db        = new Matrix<Dtype>(this->_bias);
	this->_dE_dw        = new Matrix<Dtype>(this->_w);

	this->_w_inc        = new Matrix<Dtype>(this->_w);
	this->_bias_inc     = new Matrix<Dtype>(this->_bias);

	this->_w_inc->zeros();
	this->_bias_inc->zeros();
}

template <typename Dtype>
void InnerProductLayer<Dtype>::computeOutputs(Matrix<Dtype>* x){ 
//	x->showValue("data");
//	this->_w->showValue("w");
//	x->reValue(1.0f);
//	this->_w->reValue(1.0f);
	x->rightMult(this->_w, 1, this->_y, this->handle);
	this->_y->addRowVector(this->_bias);
//	this->_y->showValue("yj1");

}


template <typename Dtype>
void InnerProductLayer<Dtype>::computeDerivsOfPars(Matrix<Dtype>* x){

	Matrix<Dtype>* data_T = new Matrix<Dtype>(x->getNumCols(), x->getNumRows());
	x->getTranspose(data_T);

	data_T->rightMult(this->_dE_dy, 1, this->_dE_dw, this->handle);
	this->_dE_dy->sumRow(this->_dE_db);

//this->_dE_dw->showValue("dedwinner");
//this->_dE_dy->showValue("innerdedy");
	delete data_T;
}

template <typename Dtype>
void InnerProductLayer<Dtype>::computeDerivsOfInput(Matrix<Dtype>* dE_dx){
	Matrix<Dtype>* w_T = new Matrix<Dtype>(this->_w->getNumCols(), this->_w->getNumRows());
	this->_w->getTranspose(w_T);
	this->_dE_dy->rightMult(w_T, 1, dE_dx, this->handle);
	delete w_T;
}

