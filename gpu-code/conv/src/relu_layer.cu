///
/// \file sigmoid_layer.cu
/// @brief


using namespace std;

template <typename Dtype>
ReluLayer<Dtype>::ReluLayer(Param* p){

	this->_p           = p;
}

template <typename Dtype>
ReluLayer<Dtype>::~ReluLayer() {
	delete  this->_y; 
	delete  this->_dE_dy;
	delete _record;
}

template <typename Dtype>
void ReluLayer<Dtype>::initCuda() {


	ConnectType ct = this->_p->getConnectType();
	int col;
	if(ct == PARAM_CONNECT_TYPE_LOCAL)
		col = _p->getOutHeight()*_p->getOutWidth() \
			  * this->_p->getOutChannel(); 
	else if(ct == PARAM_CONNECT_TYPE_FULL)
		col = this->_p->getNumOut(); 
	this->_y             = new Matrix<Dtype>(_p->getMinibatchSize(), \
								col);
	this->_dE_dy         = new Matrix<Dtype>(this->_y);
	
	_record				 = new Matrix<int>(_p->getMinibatchSize(), col);

}

template <typename Dtype>
void ReluLayer<Dtype>::computeOutput(Matrix<Dtype>* x){ 
//	x->reValue(-0.1f);
//	x->showValue("data");

	this->_y->zeros();	
	x->applyRelu(this->_y, _record);
	
//	if(_p->getName() == "relu1_layer")
//		this->_y->showValue(_p->getName() +"relu_y");
}

template <typename Dtype>
void ReluLayer<Dtype>::computeDerivsOfInput(Matrix<Dtype>* dE_dx){
	dE_dx->zeros();

//this->_dE_dy->reValue(1.0f);
//_record->reValue(0.0f);
	
	this->_dE_dy->applyRelu(dE_dx, _record, false);

//	if(_p->getName() == "relu1_layer")
//		dE_dx->showValue("dedx");
}


