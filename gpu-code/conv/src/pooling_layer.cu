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
	if(_lcp->getOutSize() > MAX_THREAD_SIZE && _overlap_len > 0)	
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
	int _overlap_len = _lcp->getFilterSize() - _lcp->getStride();
	if(_lcp->getOutSize() > MAX_THREAD_SIZE && _overlap_len > 0){	
		unranged_dE_dx = new Matrix<Dtype>(_lcp->getMinibatchSize(), \
				pow(_lcp->getBoxInSize() * _lcp->getBoxNumSize(), 2) \
				* _lcp->getOutChannel());
	}

}

template <typename Dtype>
void PoolingLayer<Dtype>::computeOutputs(Matrix<Dtype>* x){

	this->_y->zeros();	
	int num_box = pow(_lcp->getBoxNumSize(), 2);
	

	dim3 blocks = dim3(_lcp->getMinibatchSize(), _lcp->getInChannel() * num_box);
	dim3 threads = dim3(MAX_THREAD_SIZE, MAX_THREAD_SIZE); 

	/// 每个block计算输出32*32的大小
	/// 同时并行多个block
	//	x->reValue(16);
	int box_out_size = MAX_THREAD_SIZE > _lcp->getOutSize() \
					? _lcp->getOutSize() : MAX_THREAD_SIZE;
	if(_lcp->getPoolType() == MAX_POOLING ){
		
		max_pooling<<<blocks, threads, \
			sizeof(Dtype)*pow(box_out_size, 2)*pow(_lcp->getFilterSize(), \
					2)>>>(x->getDevData(), this->_y->getDevData(), \
					_max_pos->getDevData(), _lcp->getInSize(), \
					_lcp->getInChannel(), _lcp->getOutSize(), \
					_lcp->getFilterSize(), _lcp->getStride(), \
					box_out_size, _lcp->getBoxNumSize());  
//	this->_y->showValue("y");

	}else if(_lcp->getPoolType() == AVG_POOLING){
		avg_pooling<<<blocks, threads, \
			sizeof(Dtype)*pow(box_out_size, 2)*pow(_lcp->getFilterSize(), \
					2)>>>(x->getDevData(), this->_y->getDevData(), \
					_lcp->getInSize(), \
					_lcp->getInChannel(), _lcp->getOutSize(), \
					_lcp->getFilterSize(), _lcp->getStride(), \
					box_out_size, _lcp->getBoxNumSize());  
	}else{
		cout << "Pooling type is invalid !\n";	
		exit(EXIT_FAILURE);
	}

	cudaThreadSynchronize();
	cudaCheckError();
//	if(_lcp->getName() == "pool3_layer")
//		this->_y->showValue(_lcp->getName() + "y");

}

template <typename Dtype>
void PoolingLayer<Dtype>::computeDerivsOfInput(Matrix<Dtype>* dE_dx){

	int num_box = pow(_lcp->getBoxNumSize(), 2);

		/// 计算一个box的pooling对应的输入行列大小

	dim3 blocks = dim3(_lcp->getMinibatchSize(), _lcp->getInChannel() * num_box);
	dim3 threads = dim3(MAX_THREAD_SIZE, MAX_THREAD_SIZE);

	int box_out_size = MAX_THREAD_SIZE > _lcp->getOutSize() \
						? _lcp->getOutSize() : MAX_THREAD_SIZE;

	int box_in_size = MAX_THREAD_SIZE > _lcp->getOutSize() \
						  ? _lcp->getInSize() : _lcp->getBoxInSize();

	Dtype* p_dE_dx;
	if(MAX_THREAD_SIZE < _lcp->getOutSize() && _overlap_len > 0){
		unranged_dE_dx->zeros();
		p_dE_dx = unranged_dE_dx->getDevData();
	}else{
		dE_dx->zeros();
		p_dE_dx = dE_dx->getDevData();
	}

	if(_lcp->getPoolType() == MAX_POOLING ){
//	this->_dE_dy->reValue(48);
//	_max_pos->reValue(1.0f);
		compute_dE_dy_max<<<blocks, threads, \
				sizeof(Dtype)*pow(box_in_size, 2)>>>(this->_dE_dy->getDevData(), \
					p_dE_dx, _max_pos->getDevData(), box_in_size, \
					box_out_size, _lcp->getInChannel(), _lcp->getOutSize(), \
					_lcp->getFilterSize(), _lcp->getStride(), _lcp->getBoxNumSize());
		cudaThreadSynchronize();
		cudaCheckError();

		if(_lcp->getOutSize() > MAX_THREAD_SIZE && _overlap_len > 0){
	//unranged_dE_dx->showValue("unrangeddEdx");
			blocks = dim3(_lcp->getMinibatchSize(), _lcp->getInChannel());
			
			compactOverlap<<<blocks, threads, sizeof(Dtype)*pow(_lcp->getInSize(),2)>>>( \
					unranged_dE_dx->getDevData(), dE_dx->getDevData(), _lcp->getInSize(), \
					_lcp->getBoxInSize(),  _overlap_len, \
					_lcp->getBoxInSize() * _lcp->getBoxNumSize(), _lcp->getOutChannel());
			cudaThreadSynchronize();
			cudaCheckError();
		}
//	dE_dx->showValue("dEdx");

	}else if(_lcp->getPoolType() == AVG_POOLING){
//	this->_dE_dy->reValue(50);
		compute_dE_dy_avg<<<blocks, threads, \
				sizeof(Dtype)*pow(box_in_size, 2)>>>(this->_dE_dy->getDevData(), \
					p_dE_dx, box_in_size, box_out_size, \
					_lcp->getInChannel(), \
					_lcp->getOutSize(), _lcp->getFilterSize(), \
					_lcp->getStride(), _lcp->getBoxNumSize());
		cudaThreadSynchronize();
		cudaCheckError();

		if(_lcp->getOutSize() > MAX_THREAD_SIZE && _overlap_len > 0){
			blocks = dim3(_lcp->getMinibatchSize(), _lcp->getInChannel());
			
//	unranged_dE_dx->showValue("unrangeddEdx");
			compactOverlap<<<blocks, threads, sizeof(Dtype)*pow(_lcp->getInSize(),2)>>>( \
					unranged_dE_dx->getDevData(), dE_dx->getDevData(), _lcp->getInSize(), \
					_lcp->getBoxInSize(),  _overlap_len, \
					_lcp->getBoxInSize() * _lcp->getBoxNumSize(), _lcp->getOutChannel());
			cudaThreadSynchronize();
			cudaCheckError();
		}
//	dE_dx->showValue("dEdx");

	}else{
		cout << "Pooling type is invalid !\n";	
		exit(EXIT_FAILURE);
	}

}



