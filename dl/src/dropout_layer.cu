///
/// \file dropout_layer.cu
/// @brief


using namespace std;

template <typename Dtype>
DropoutLayer<Dtype>::DropoutLayer(Param* p){

	this->_p           = p;
}

template <typename Dtype>
DropoutLayer<Dtype>::~DropoutLayer() {
	delete  this->_y; 
	delete  this->_dE_dy;
	delete  _drop_record;
	delete  _drop_rand_probs;

}

template <typename Dtype>
void DropoutLayer<Dtype>::initCuda() {


	ConnectType ct = this->_p->getConnectType();
	int col;
	if(ct == PARAM_CONNECT_TYPE_LOCAL)
		col = _p->getOutHeight()*_p->getOutWidth() \
			  * this->_p->getOutChannel(); 
	else if(ct == PARAM_CONNECT_TYPE_FULL)
		col = this->_p->getNumOut();
		
	this->_y             = new Matrix<Dtype>(_p->getMinibatchSize(), col);
	this->_dE_dy         = new Matrix<Dtype>(this->_y);
	_drop_record		 = new Matrix<int>(_p->getMinibatchSize(), col);
	_drop_rand_probs     = new Matrix<curandState>(_p->getMinibatchSize(), col);
	_is_set_up 			 = false;
}

template <typename Dtype>
void DropoutLayer<Dtype>::computeOutput(Matrix<Dtype>* x){ 
	
	x->applyDropout(this->_y, _drop_record, _drop_rand_probs, _is_set_up);
	
	if(_is_set_up == false)
		_is_set_up = true;
	

}

template <typename Dtype>
void DropoutLayer<Dtype>::computeDerivsOfInput(Matrix<Dtype>* dE_dx){

	this->_dE_dy->applyRelu(dE_dx, _drop_record, false);

}


