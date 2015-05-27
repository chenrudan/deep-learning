/*
 * filename: inner_product_layer.cu
 */

#include <time.h>
#include "inner_product_layer.cuh"
#include "layer_kernel.cuh"

using namespace std;

InnerProductLayer::InnerProductLayer(pars* network){
	this->_num_in                   = network->num_in;
	this->_num_out                  = network->num_out;

	//w_hk的learning rate
	this->_w_lr                     = network->w_lr;
	//out bias learning rate
	this->_b_lr                     = network->b_lr;
	//上一次更新的参数控制增长趋势
	this->_momentum                 = network->momentum;
	this->_weight_decay             = network->weight_decay;

	this->_minibatch_size           = network->minibatch_size;
	this->_lr_down_scale            = network->lr_down_scale;

	cublasCreate(&handle);
}

InnerProductLayer::~InnerProductLayer() {

	delete _w; 
	delete _w_inc;
	delete _bias;
	delete _bias_inc;

	delete  _y; 
	delete  _dE_dy;
	delete _dE_db;
	delete _dE_dw;
	delete _dE_dx_sigmoid;
	cublasDestroy(handle);
}

void InnerProductLayer::initCuda() {

	this->_w            = new NVMatrix(_num_in, _num_out);
	//                  NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	this->_bias         = new NVMatrix(1, _num_out);
	//                  NVMatrix::ALLOC_ON_UNIFIED_MEMORY);

	this->_y               = new NVMatrix(_minibatch_size, _num_out);

	this->_dE_dx_sigmoid	= new NVMatrix(_minibatch_size, _num_out);	
	this->_dE_dy           = new NVMatrix(_y);
	this->_dE_db           = new NVMatrix(_bias);
	this->_dE_dw          = new NVMatrix(_w);

	this->_w_inc         = new NVMatrix(_w);
	this->_bias_inc        = new NVMatrix(1, _num_out);
	this->_w_inc->zeros();
	this->_bias_inc->zeros();
}

void InnerProductLayer::computeOutputs(NVMatrix* x){ 
//	x->showValue("data");
//	_w->showValue("w");
	x->rightMult(_w, 1, _y, handle);
	_y->addRowVector(_bias);
	_y->apply(NVMatrix::SIGMOID);
//	_y->showValue("yj1");
}

void InnerProductLayer::computeDerivsOfPars(NVMatrix* x){

	_y->subtractFromScalar(1, _dE_dx_sigmoid);

	_dE_dx_sigmoid->eltWiseMult(_y);

	_dE_dx_sigmoid->eltWiseMult(_dE_dy);

	NVMatrix* data_T = new NVMatrix(x->getNumCols(), x->getNumRows());
	x->getTranspose(data_T);

	data_T->rightMult(_dE_dx_sigmoid, 1, _dE_dw, handle);
	_dE_dx_sigmoid->sumRow(_dE_db);

//_dE_dw->showValue("dedwinner");
	delete data_T;
}

void InnerProductLayer::computeDerivsOfInput(NVMatrix* dE_dx){
	NVMatrix* w_T = new NVMatrix(_w->getNumCols(), _w->getNumRows());
	_w->getTranspose(w_T);
	_dE_dx_sigmoid->rightMult(w_T, 1, dE_dx, handle);
	delete w_T;
}


