///
/// \file pooling_layer.cu
///

#include "pooling_layer.hpp"

using namespace std;

template <typename Dtype>
PoolingLayer<Dtype>::PoolingLayer(LocalConnectParam *lcp){
	this->_lcp = lcp;

	cublasCreate(&this->handle);
}

template <typename Dtype>
PoolingLayer<Dtype>::~PoolingLayer() {

	delete this-> _y;
	delete this->_dE_dy;

	cudaFree(_max_pos);
	cublasDestroy(this->handle);
}

template <typename Dtype>
void PoolingLayer<Dtype>::initCuda() {


	this->_y               = new Matrix<Dtype>(_lcp->getMinibatchSize(), \
								pow(_lcp->getOutSize(), 2) * _lcp->getOutChannel());

	this->_dE_dy           = new Matrix<Dtype>(this->_y);


	cudaError_t status;
	status = cudaMalloc((void**) &_max_pos, \
                _lcp->getMinibatchSize() * _lcp->getInChannel() \
                       * pow(_lcp->getOutSize(), 2) * sizeof(int));
	if (status != cudaSuccess) {
		fprintf(stderr, "!!!! device memory allocation error\n");
		exit(EXIT_FAILURE);
	}
}

template <typename Dtype>
void PoolingLayer<Dtype>::computeOutputs(Matrix<Dtype>* x){
	dim3 blocks = dim3(_lcp->getMinibatchSize(), _lcp->getInChannel());
	dim3 threads = dim3(ceil(_lcp->getOutSize() / 16.0) * 16,  \
			ceil(_lcp->getOutSize() / 16.0) * 16);

//	x->reValue(32);

	max_pooling<<<blocks, threads, \
		sizeof(Dtype)*pow(_lcp->getOutSize(), 2)*pow(_lcp->getFilterSize(), 2)>>>(x->getDevData(), \
			this->_y->getDevData(), _max_pos, _lcp->getInSize(), \
			_lcp->getOutSize(), _lcp->getFilterSize(), _lcp->getStride());  
	cudaThreadSynchronize();

//x->showValue("x");
//_y->showValue("y");
}

template <typename Dtype>
void PoolingLayer<Dtype>::computeDerivsOfInput(Matrix<Dtype>* dE_dx){

	dim3 blocks = dim3(_lcp->getMinibatchSize(), \
			_lcp->getInChannel());
	dim3 threads = dim3(ceil(_lcp->getOutSize() / 16.0) * 16,  \
			ceil(_lcp->getOutSize() / 16.0) * 16);

	dE_dx->zeros();
//_dE_dy->reValue(16);
	
/*
int length = dE_dx->getNumRows()*dE_dx->getNumCols();
int* tmp = new int[length];
cudaMemcpy(tmp, _max_pos, sizeof(int)*length, cudaMemcpyDeviceToHost);
cout << "maxpos\n";
for(int i = 0; i < length; i++){
	cout << tmp[i] << " ";
}*/

	compute_dE_dy_max<<<blocks, threads, \
		sizeof(Dtype)*_lcp->getInSize()*_lcp->getInSize()>>>(this->_dE_dy->getDevData(), \
			dE_dx->getDevData(), _max_pos, _lcp->getInSize(), \
			_lcp->getOutSize(), _lcp->getFilterSize(), _lcp->getStride());
	cudaThreadSynchronize();
//dE_dx->showValue("dEdx");
}

/*
   void ConvNet::computeAvgOutputs(){
	   //16*16
	   dim3 blocks = dim3(_lcp->getMinibatchSize(), _lcp->getInChannel()s);
	   dim3 threads = dim3(_lcp->getOutSize(), _lcp->getOutSize());
	   //24*24,pooling到12*12
	   avg_pooling<<<blocks, threads>>>(_y_h->getDevData(), _y_i->getDevData());
	   cudaThreadSynchronize();
   }*/




