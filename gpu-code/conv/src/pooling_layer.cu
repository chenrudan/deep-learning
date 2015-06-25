///
/// \file pooling_layer.cu
///

#include "pooling_layer.hpp"

using namespace std;

template <typename Dtype>
PoolingLayer<Dtype>::PoolingLayer(PoolParam *lcp){
	this->_lcp = lcp;

	cublasCreate(&this->handle);
}

template <typename Dtype>
PoolingLayer<Dtype>::~PoolingLayer() {

	delete this-> _y;
	delete this->_dE_dy;

	if(_lcp->getPoolType() == MAX_POOLING )
		cudaFree(_max_pos);
	cublasDestroy(this->handle);
}

template <typename Dtype>
void PoolingLayer<Dtype>::initCuda() {


	this->_y               = new Matrix<Dtype>(_lcp->getMinibatchSize(), \
			pow(_lcp->getOutSize(), 2) * _lcp->getOutChannel());

	this->_dE_dy           = new Matrix<Dtype>(this->_y);


	if(_lcp->getPoolType() == MAX_POOLING ){
		cudaError_t status;
		status = cudaMalloc((void**) &_max_pos, \
				_lcp->getMinibatchSize() * _lcp->getInChannel() \
				* pow(_lcp->getOutSize(), 2) * sizeof(int));
		if (status != cudaSuccess) {
			fprintf(stderr, "!!!! device memory allocation error\n");
			exit(EXIT_FAILURE);
		}
	}
}

template <typename Dtype>
void PoolingLayer<Dtype>::computeOutputs(Matrix<Dtype>* x){
	dim3 blocks = dim3(_lcp->getMinibatchSize(), _lcp->getInChannel());
	dim3 threads = dim3(ceil(_lcp->getOutSize() / 16.0) * 16,  \
			ceil(_lcp->getOutSize() / 16.0) * 16);

	//	x->reValue(32);

	if(_lcp->getPoolType() == MAX_POOLING )
		max_pooling<<<blocks, threads, \
			sizeof(Dtype)*pow(_lcp->getOutSize(), 2)*pow(_lcp->getFilterSize(), 2)>>>(x->getDevData(), \
					this->_y->getDevData(), _max_pos, _lcp->getInSize(), \
					_lcp->getOutSize(), _lcp->getFilterSize(), _lcp->getStride());  
	else if(_lcp->getPoolType() == AVG_POOLING)
		avg_pooling<<<blocks, threads, \
			sizeof(Dtype)*pow(_lcp->getOutSize(), 2)*pow(_lcp->getFilterSize(), 2)>>>(x->getDevData(), \
					this->_y->getDevData(), _lcp->getInSize(), _lcp->getOutSize(), \
					_lcp->getFilterSize(), _lcp->getStride());  
	else{
		cout << "Pooling type is invalid !\n";	
		exit(EXIT_FAILURE);
	}

	cudaThreadSynchronize();

	//x->showValue("x");
	//_y->showValue("y");
}

template <typename Dtype>
void PoolingLayer<Dtype>::computeDerivsOfInput(Matrix<Dtype>* dE_dx){

	dE_dx->zeros();
	dim3 blocks = dim3(_lcp->getMinibatchSize(), \
			_lcp->getInChannel());
	dim3 threads = dim3(ceil(_lcp->getOutSize() / 16.0) * 16,  \
			ceil(_lcp->getOutSize() / 16.0) * 16);
//	this->_dE_dy->reValue(9.0f);
	if(_lcp->getPoolType() == MAX_POOLING ){
		compute_dE_dy_max<<<blocks, threads, \
			sizeof(Dtype)*_lcp->getInSize()*_lcp->getInSize()>>>(this->_dE_dy->getDevData(), \
					dE_dx->getDevData(), _max_pos, _lcp->getInSize(), \
					_lcp->getOutSize(), _lcp->getFilterSize(), _lcp->getStride());
	}else if(_lcp->getPoolType() == AVG_POOLING){
		compute_dE_dy_avg<<<blocks, threads, \
			sizeof(Dtype)*_lcp->getInSize()*_lcp->getInSize()>>>(this->_dE_dy->getDevData(), \
					dE_dx->getDevData(), _lcp->getInSize(), \
					_lcp->getOutSize(), _lcp->getFilterSize(), _lcp->getStride());

	}else{
		cout << "Pooling type is invalid !\n";	
		exit(EXIT_FAILURE);
	}


	cudaThreadSynchronize();
//	dE_dx->showValue("dEdx");
}



