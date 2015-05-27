/*
 * filename: logistic.cu
 */
#include <time.h>
#include "logistic.cuh"
#include "layer_kernel.cuh"

using namespace std;

Logistic::Logistic(pars* network){
	this->_num_in				 	= network->num_in;
	this->_num_out               	= network->num_out;

	//w_hk的learning rate
	this->_w_lr		      	      	= network->w_lr;
	//out bias learning rate
	this->_b_lr 		          	= network->b_lr;
	//上一次更新的参数控制增长趋势
	this->_momentum                 = network->momentum;
	this->_weight_decay             = network->weight_decay;

	this->_minibatch_size         	= network->minibatch_size;
	this->_lr_down_scale			= network->lr_down_scale;

	cublasCreate(&handle);
}

Logistic::~Logistic() {

	delete _w;
	delete _w_inc;
	delete _bias;
	delete _bias_inc;

	delete  _y;
	delete  _dE_dy;
	delete _dE_db;
	delete _dE_dw;
	cublasDestroy(handle);
}

void Logistic::initCuda() {

	this->_w            = new NVMatrix(_num_in, _num_out);
//					NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	this->_bias         = new NVMatrix(1, _num_out);
//					NVMatrix::ALLOC_ON_UNIFIED_MEMORY);

	this->_y               = new NVMatrix(_minibatch_size, _num_out);

	this->_dE_dy           = new NVMatrix(_y);
	this->_dE_db           = new NVMatrix(_bias);
	this->_dE_dw          = new NVMatrix(_w);

	this->_w_inc         = new NVMatrix(_w);
	this->_bias_inc        = new NVMatrix(1, _num_out);
	this->_w_inc->zeros();
	this->_bias_inc->zeros();
}

void Logistic::computeOutputs(NVMatrix* x){
//x->showValue("data");
	x->rightMult(_w, 1, _y, handle);
//_w->showValue("w");
//_y->showValue("yj1");
	_y->addRowVector(_bias);
	_y->apply(NVMatrix::SOFTMAX);
}

double Logistic::computeError(NVMatrix* labels, int& num_error){

	Matrix* h_labels = new Matrix(labels->getNumRows(), labels->getNumCols());
	labels->copyToHost(h_labels);

	Matrix* y_CPU = new Matrix(_y->getNumRows(), _y->getNumCols());
	_y->copyToHost(y_CPU);
	Matrix* correct_probs = new Matrix(_y->getNumRows(), 1);
	NVMatrix* d_max_pos_of_out = new NVMatrix(_y->getNumRows(), 1);
	_y->maxPosInRow(d_max_pos_of_out);
	Matrix* h_max_pos_of_out = new Matrix(_y->getNumRows(), 1);
	d_max_pos_of_out->copyToHost(h_max_pos_of_out);

	for (int c = 0; c < _y->getNumRows(); c++) {
		int true_label = h_labels->getCell(c, 0);
		int predict_label = h_max_pos_of_out->getCell(c, 0);
		correct_probs->getCell(c, 0) = y_CPU->getCell(c, true_label);

//cout << predictLabel << ":" << trueLabel << " ";
		if(predict_label != true_label)
			num_error++;
	}
//cout << endl;
	correct_probs->apply(Matrix::LOG);
	double result = -correct_probs->sum();
	cudaThreadSynchronize();

	delete h_labels;
	delete y_CPU;
	delete correct_probs;
	delete d_max_pos_of_out;
	delete h_max_pos_of_out;
	return result;
}

void Logistic::computeDerivsOfPars(NVMatrix* x, NVMatrix* labels){
	assert(labels->getNumRows() == x->getNumRows());

	const int num_thread = DIVUP(_num_out, ADD_BLOCK_SIZE) * ADD_BLOCK_SIZE;
	compute_dE_dy<<<_minibatch_size, num_thread>>>(_y->getDevData(), \
			labels->getDevData(), _dE_dy->getDevData(), _num_out);

	NVMatrix* data_T = new NVMatrix(x->getNumCols(), x->getNumRows());
	x->getTranspose(data_T);

	data_T->rightMult(_dE_dy, 1, _dE_dw, handle);
//	_y->showValue("ysoftmax");
	_dE_dy->sumRow(_dE_db);

	delete data_T;
}

void Logistic::computeDerivsOfInput(NVMatrix* dE_dx){
	NVMatrix* w_T = new NVMatrix(_w->getNumCols(), _w->getNumRows());
	_w->getTranspose(w_T);
	_dE_dy->rightMult(w_T, 1, dE_dx, handle);
	delete w_T;
}



