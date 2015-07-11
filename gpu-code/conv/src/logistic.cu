///
/// \file logistic.cu
///

#include "logistic.hpp"

using namespace std;

template <typename Dtype>
Logistic<Dtype>::Logistic<Dtype>(FullConnectParam* fcp) {
	this->_fcp = fcp;
	
}

template <typename Dtype>
Logistic<Dtype>::~Logistic<Dtype>() {

	delete this->_y;
	delete[] h_labels;
	delete[] y_CPU;
	delete[] correct_probs;
	delete d_max_pos_of_out;
	delete[] h_max_pos_of_out;
	delete _d_record;
	delete[] _h_record;
}

template <typename Dtype>
void Logistic<Dtype>::initCuda() {

	this->_y            = new Matrix<Dtype>(this->_fcp->getMinibatchSize(), \
								this->_fcp->getNumOut());
	h_labels 			= new Dtype[this->_fcp->getMinibatchSize()];
	y_CPU 				= new Dtype[this->_y->getNumEles()];
	correct_probs 		= new Dtype[this->_y->getNumRows()];
	d_max_pos_of_out 	= new Matrix<Dtype>(this->_y->getNumRows(), 1);
	h_max_pos_of_out 	= new Dtype[this->_y->getNumRows()];

	_d_record 		= new Matrix<int>(this->_y->getNumCols(), this->_y->getNumCols());
	_h_record 		= new int[this->_y->getNumCols() * this->_y->getNumCols()];
}

template <typename Dtype>
void Logistic<Dtype>::computeOutputs(Matrix<Dtype>* x){
//x->showValue("data");
	x->apply(Matrix<Dtype>::SOFTMAX, this->_y);
//this->_y->showValue("yj1");
}

template <typename Dtype>
double Logistic<Dtype>::computeError(Matrix<Dtype>* labels, int& num_error){

	/// h_labels大小是minibatch * 1
	labels->copyToHost(h_labels, labels->getNumEles());

	/// y_cpu大小是minibatch * 10
	this->_y->copyToHost(y_CPU, this->_y->getNumEles());

	/// 记录找打的最大位置上的likelihood
	/// 记录最大位置的下标
	this->_y->maxPosInRow(d_max_pos_of_out);
//d_max_pos_of_out->showValue("maxpos");
//this->_y->showValue("yj1");

	d_max_pos_of_out->copyToHost(h_max_pos_of_out, this->_y->getNumRows());

	for (int c = 0; c < this->_y->getNumRows(); c++) {
		int true_label = h_labels[c];
		int predict_label = h_max_pos_of_out[c];
		correct_probs[c] = log(y_CPU[c * this->_y->getNumCols() + true_label]);

		if(predict_label != true_label)
			num_error++;
		_h_record[predict_label * this->_y->getNumCols() + true_label]++ ;
	}
	double result = 0;
	for(int i = 0; i < labels->getNumEles(); i++){
		result -= correct_probs[i];
	}

	return result;
}

template <typename Dtype>
void Logistic<Dtype>::computeDerivsOfInput(Matrix<Dtype>* dE_dx, Matrix<Dtype>* labels){
	assert(labels->getNumRows() == dE_dx->getNumRows());

//this->_y->reValue(1.0f);
//labels->reValue(1.0f);

	const int num_thread = DIVUP(this->_fcp->getNumOut(), ADD_BLOCK_SIZE) * ADD_BLOCK_SIZE;
	compute_dE_dy<<<this->_fcp->getMinibatchSize(), num_thread>>>(this->_y->getDevData(), \
			labels->getDevData(), dE_dx->getDevData(), this->_fcp->getNumOut());


}



