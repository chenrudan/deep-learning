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
		delete _max_pos;
	if(_lcp->getOutSize() > MAX_THREAD_SIZE)	
		delete unranged_dE_dx;
	cublasDestroy(this->handle);
}

template <typename Dtype>
void PoolingLayer<Dtype>::initCuda() {


	this->_y               = new Matrix<Dtype>(_lcp->getMinibatchSize(), \
			pow(_lcp->getOutSize(), 2) * _lcp->getOutChannel());

	this->_dE_dy           = new Matrix<Dtype>(this->_y);


	if(_lcp->getPoolType() == MAX_POOLING ){
		_max_pos           = new Matrix<int>(_lcp->getMinibatchSize(), \
			pow(_lcp->getOutSize(), 2) * _lcp->getOutChannel());
	
	}
	if(_lcp->getOutSize() > MAX_THREAD_SIZE){	
		unranged_dE_dx = new Matrix<Dtype>(_lcp->getMinibatchSize(), \
				pow(_lcp->getBoxInSize() * _lcp->getBoxNumSize(), 2) \
				* _lcp->getOutChannel());
	}

}

template <typename Dtype>
void PoolingLayer<Dtype>::computeOutputs(Matrix<Dtype>* x){
	
	int num_box = pow(_lcp->getBoxNumSize(), 2);
	

	dim3 blocks = dim3(_lcp->getMinibatchSize(), _lcp->getInChannel() * num_box);
	dim3 threads = dim3(MAX_THREAD_SIZE, MAX_THREAD_SIZE); 
//		x->reValue(32);

	/// 每个block计算输出32*32的大小
	/// 同时并行多个block
	if(_lcp->getPoolType() == MAX_POOLING ){
		max_pooling<<<blocks, threads, \
			sizeof(Dtype)*pow(MAX_THREAD_SIZE, 2)*pow(_lcp->getFilterSize(), 2)>>>(x->getDevData(), \
					this->_y->getDevData(), _max_pos->getDevData(), _lcp->getInSize(), \
					_lcp->getInChannel(), _lcp->getOutSize(), \
					_lcp->getFilterSize(), _lcp->getStride(), _lcp->getBoxNumSize());  

	}else if(_lcp->getPoolType() == AVG_POOLING)
		avg_pooling<<<blocks, threads, \
			sizeof(Dtype)*pow(MAX_THREAD_SIZE, 2)*pow(_lcp->getFilterSize(), 2)>>>(x->getDevData(), \
					this->_y->getDevData(), _lcp->getInSize(), \
					_lcp->getInChannel(), _lcp->getOutSize(), \
					_lcp->getFilterSize(), _lcp->getStride(), _lcp->getBoxNumSize());  
	else{
		cout << "Pooling type is invalid !\n";	
		exit(EXIT_FAILURE);
	}

	cudaThreadSynchronize();
//	x->showValue("x");
//	this->_y->showValue("y");

}

template <typename Dtype>
void PoolingLayer<Dtype>::computeDerivsOfInput(Matrix<Dtype>* dE_dx){

	dE_dx->zeros();
	int num_box = pow(_lcp->getBoxNumSize(), 2);

		/// 计算一个box的pooling对应的输入行列大小

	dim3 blocks = dim3(_lcp->getMinibatchSize(), _lcp->getInChannel() * num_box);
	dim3 threads = dim3(MAX_THREAD_SIZE, MAX_THREAD_SIZE);


	if(_lcp->getPoolType() == MAX_POOLING ){
	this->_dE_dy->reValue(16);
		_max_pos->reValue(1.0f);
		_max_pos->showValue("maxpos");

		if(_lcp->getOutSize() > MAX_THREAD_SIZE){
			compute_dE_dy_max<<<blocks, threads, \
				sizeof(Dtype)*pow(_lcp->getBoxInSize(), 2)>>>(this->_dE_dy->getDevData(), \
					unranged_dE_dx->getDevData(), _max_pos->getDevData(), _lcp->getBoxInSize(), \
					MAX_THREAD_SIZE, _lcp->getInChannel(), _lcp->getOutSize(), \
					_lcp->getFilterSize(), _lcp->getStride(), _lcp->getBoxNumSize());
		}else{
			compute_dE_dy_max<<<blocks, threads, \
				sizeof(Dtype)*pow(_lcp->getInSize(), 2)>>>(this->_dE_dy->getDevData(), \
					dE_dx->getDevData(), _max_pos->getDevData(), _lcp->getInSize(), \
					_lcp->getOutSize(), _lcp->getInChannel(), _lcp->getOutSize(), \
					_lcp->getFilterSize(), _lcp->getStride(), _lcp->getBoxNumSize());
			dE_dx->showValue("dEdx");
		}

	}else if(_lcp->getPoolType() == AVG_POOLING){
		if(_lcp->getOutSize() > MAX_THREAD_SIZE){
		
		
		}else{
			compute_dE_dy_avg<<<blocks, threads, \
				sizeof(Dtype)*_lcp->getInSize()*_lcp->getInSize()>>>(this->_dE_dy->getDevData(), \
					dE_dx->getDevData(), _lcp->getInSize(), _lcp->getOutSize(), _lcp->getInChannel(), \
					_lcp->getOutSize(), _lcp->getFilterSize(), \
					_lcp->getStride(), _lcp->getBoxNumSize());
		}

	}else{
		cout << "Pooling type is invalid !\n";	
		exit(EXIT_FAILURE);
	}
	//this->_dE_dy->showValue("dEdy");

	cudaThreadSynchronize();
}



