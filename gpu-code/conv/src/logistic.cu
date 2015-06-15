///
/// \file logistic.cu
///

#include "logistic.hpp"

using namespace std;

template <typename Dtype>
Logistic<Dtype>::Logistic<Dtype>(FullConnectParam* fcp) : \
 	TrainLayer<Dtype>((TrainParam*)fcp) {
	this->_fcp = fcp;
	cublasCreate(&this->handle);
}

template <typename Dtype>
Logistic<Dtype>::~Logistic<Dtype>() {

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
void Logistic<Dtype>::initCuda() {

	this->_w            = new Matrix<Dtype>(this->_fcp->getNumIn(), this->_fcp->getNumOut());
	this->_bias         = new Matrix<Dtype>(1, this->_fcp->getNumOut());

	this->_y            = new Matrix<Dtype>(this->_fcp->getMinibatchSize(), this->_fcp->getNumOut());

	this->_dE_dy        = new Matrix<Dtype>(this->_y);
	this->_dE_db        = new Matrix<Dtype>(this->_bias);
	this->_dE_dw        = new Matrix<Dtype>(this->_w);

	this->_w_inc        = new Matrix<Dtype>(this->_w);
	this->_bias_inc     = new Matrix<Dtype>(1, this->_fcp->getNumOut());
	
	this->_w_inc->zeros();
	this->_bias_inc->zeros();
}

template <typename Dtype>
void Logistic<Dtype>::computeOutputs(Matrix<Dtype>* x){
//x->showValue("data");
	x->rightMult(this->_w, 1, this->_y, this->handle);
//this->_w->showValue("w");
//this->_y->showValue("yj1");
	this->_y->addRowVector(this->_bias);
	this->_y->apply(Matrix<Dtype>::SOFTMAX);
}

template <typename Dtype>
double Logistic<Dtype>::computeError(Matrix<Dtype>* labels, int& num_error){


	/// h_labels大小是minibatch * 1
	Dtype* h_labels = new Dtype[labels->getNumEles()];
	labels->copyToHost(h_labels, labels->getNumEles());

	/// y_cpu大小是minibatch * 10
	Dtype* y_CPU = new Dtype(this->_y->getNumEles());
	this->_y->copyToHost(y_CPU, this->_y->getNumEles());

	/// 记录找打的最大位置上的likelihood
	Dtype* correct_probs = new Dtype[this->_y->getNumRows()];
	/// 记录最大位置的下标
	Matrix<Dtype>* d_max_pos_of_out = new Matrix<Dtype>(this->_y->getNumRows(), 1);
	this->_y->maxPosInRow(d_max_pos_of_out);

	Dtype* h_max_pos_of_out = new float[this->_y->getNumRows()];
	d_max_pos_of_out->copyToHost(h_max_pos_of_out, this->_y->getNumRows());

	for (int c = 0; c < this->_y->getNumRows(); c++) {
		int true_label = h_labels[c];
		int predict_label = h_max_pos_of_out[c];
		correct_probs[c] = log(y_CPU[c * this->_y->getNumCols() + true_label]);

//cout << predictLabel << ":" << trueLabel << " ";
		if(predict_label != true_label)
			num_error++;
	}
//cout << endl;
	double result = 0;
	for(int i = 0; i < labels->getNumEles(); i++){
		result -= correct_probs[i];
	}

	delete h_labels;
	delete y_CPU;
	delete correct_probs;
	delete d_max_pos_of_out;
	delete h_max_pos_of_out;
	return result;
}

template <typename Dtype>
void Logistic<Dtype>::computeDerivsOfPars(Matrix<Dtype>* x, Matrix<Dtype>* labels){
	assert(labels->getNumRows() == x->getNumRows());

	const int num_thread = DIVUP(this->_fcp->getNumOut(), ADD_BLOCK_SIZE) * ADD_BLOCK_SIZE;
	compute_dE_dy<<<this->_fcp->getMinibatchSize(), num_thread>>>(this->_y->getDevData(), \
			labels->getDevData(), this->_dE_dy->getDevData(), this->_fcp->getNumOut());

	Matrix<Dtype>* data_T = new Matrix<Dtype>(x->getNumCols(), x->getNumRows());
	x->getTranspose(data_T);

	data_T->rightMult(this->_dE_dy, 1, this->_dE_dw, this->handle);
//	this->_y->showValue("ysoftmax");
	this->_dE_dy->sumRow(this->_dE_db);

	delete data_T;
}

template <typename Dtype>
void Logistic<Dtype>::computeDerivsOfInput(Matrix<Dtype>* dE_dx){
	Matrix<Dtype>* w_T = new Matrix<Dtype>(this->_w->getNumCols(), this->_w->getNumRows());
	this->_w->getTranspose(w_T);
	this->_dE_dy->rightMult(w_T, 1, dE_dx, this->handle);
	delete w_T;
}



