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


	cudaFree(_record);
}

template <typename Dtype>
void ReluLayer<Dtype>::initCuda() {


	ConnectType ct = this->_p->getConnectType();
	int col;
	if(ct == PARAM_CONNECT_TYPE_LOCAL)
		col = pow(this->_p->getOutSize(), 2) * this->_p->getOutChannel(); 
	else if(ct == PARAM_CONNECT_TYPE_FULL)
		col = this->_p->getNumOut(); 
	this->_y             = new Matrix<Dtype>(_p->getMinibatchSize(), \
								col);
	this->_dE_dy         = new Matrix<Dtype>(this->_y);

	cudaError_t status;
	status = cudaMalloc((void**) &_record, \
                this->_p->getMinibatchSize() * col * sizeof(int));
	if (status != cudaSuccess) {
		fprintf(stderr, "!!!! device memory allocation error\n");
		exit(EXIT_FAILURE);
	}
}

template <typename Dtype>
void ReluLayer<Dtype>::computeOutputs(Matrix<Dtype>* x){ 
//	x->reValue(96);
//	x->showValue("data");
	
	x->applyRelu(this->_y, _record);
	
//	this->_y->showValue("yj1");
}

template <typename Dtype>
void ReluLayer<Dtype>::computeDerivsOfInput(Matrix<Dtype>* dE_dx){

//this->_dE_dy->reValue(1.0f);
	
	this->_dE_dy->applyRelu(dE_dx, _record, false);

//dE_dx->showValue("SIGMOID_dedx");
}


