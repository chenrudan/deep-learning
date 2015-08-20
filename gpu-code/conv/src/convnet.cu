///
/// \file convnet.cu
/// @brief


#include <time.h>

#include "convnet.hpp"
#include "layer_kernel.cuh"

using namespace std;

template <typename Dtype>
ConvNet<Dtype>::ConvNet(ConvParam* cp) : TrainLayer<Dtype>(cp){

	this->_cp = cp;
	this->_filt_pixs			= pow(this->_cp->getFilterSize(), 2);
	this->_conv_pixs			= pow(this->_cp->getOutSize(), 2);
	this->_padded_in_pixs		= pow(this->_cp->getPaddedInSize(), 2);
	cublasCreate(&this->handle);
}

template <typename Dtype>
ConvNet<Dtype>::~ConvNet() {

	delete this->_w;
	delete this->_w_inc;
	delete this->_bias;
	delete this->_bias_inc;

	delete this->_y;
	delete this->_dE_dy;
	delete this->_dE_dw;
	delete this->_dE_db;
	
	delete ranged_dE_dy2;
	delete unrolled_dE_db_tmp;
	delete dE_db_tmp;
	if(this->_cp->getPad() > 0)
		delete padded_x;
	if(_cp->getOutSize() > MAX_THREAD_SIZE && _overlap_len > 0)	
		delete unranged_dE_dx;
	if(_cp->getOutSize() > MAX_THREAD_SIZE)	
		delete unranged_dE_dw;

	cublasDestroy(this->handle);
}

template <typename Dtype>
void ConvNet<Dtype>::initCuda() {

	this->_w            	= new Matrix<Dtype>(_filt_pixs \
									* this->_cp->getInChannel(), \
									this->_cp->getOutChannel());
	this->_bias         	= new Matrix<Dtype>(1, this->_cp->getOutChannel());
	this->_y            	= new Matrix<Dtype>(this->_cp->getMinibatchSize(), \
									this->_cp->getOutChannel() * _conv_pixs);
	this->_dE_dy        	= new Matrix<Dtype>(this->_y);

	this->_dE_dw          	= new Matrix<Dtype>(this->_w);
	this->_dE_db           	= new Matrix<Dtype>(this->_bias);

	this->_w_inc		 	= new Matrix<Dtype>(this->_w);
	this->_bias_inc		 	= new Matrix<Dtype>(this->_bias);

	if(this->_cp->getPad() > 0)
		this->padded_x 		= new Matrix<Dtype>(this->_cp->getMinibatchSize(), \
									this->_cp->getInChannel() * _padded_in_pixs);
	
	_overlap_len = _cp->getFilterSize() - _cp->getStride();
	if(_cp->getOutSize() > MAX_THREAD_SIZE && _overlap_len > 0){	
		unranged_dE_dx = new Matrix<Dtype>(_cp->getMinibatchSize(), \
				pow(_cp->getBoxInSize() * _cp->getBoxNumSize(), 2) \
				* _cp->getOutChannel());
	}
	if(_cp->getOutSize() > MAX_THREAD_SIZE){	
		unranged_dE_dw = new Matrix<Dtype>(pow(_cp->getFilterSize(),2) \
				*_cp->getInChannel()*pow(_cp->getBoxNumSize(),2), \
				_cp->getOutChannel());
	}
	///>变换排列方式用来做矩阵乘法计算卷积

	ranged_dE_dy2 			= new Matrix<Dtype>(this->_cp->getMinibatchSize() \
									* this->_cp->getOutChannel(), _conv_pixs);
	unrolled_dE_db_tmp 		= new Matrix<Dtype>(this->_cp->getMinibatchSize() \
									* this->_cp->getOutChannel(), 1);
	dE_db_tmp 				= new Matrix<Dtype>(this->_cp->getMinibatchSize(), \
									this->_cp->getOutChannel());

	this->_w_inc->zeros();
	this->_bias_inc->zeros();
}

template <typename Dtype>
void ConvNet<Dtype>::computeOutput(Matrix<Dtype>* x){

	this->_y->zeros();

	int num_kernel;
	int num_block;

//	x->reValue(1.0f);
//	this->_w->reValue(75);
//	this->_bias->reValue(2.0f);

	if(this->_cp->getPad() > 0){
		num_kernel = this->_cp->getMinibatchSize() * _padded_in_pixs \
					 * this->_cp->getInChannel();
		num_block = MAX_NUM_KERNEL < (num_kernel / MAX_NUM_THREAD + 1) \
					? MAX_NUM_KERNEL : (num_kernel / MAX_NUM_THREAD + 1);
		cudaMemset(padded_x->getDevData(), 0, sizeof(Dtype) * num_kernel);
		ori_to_padding<<<num_block, MAX_NUM_THREAD>>>(x->getDevData(), \
				padded_x->getDevData(), num_kernel, this->_cp->getInSize(), \
				this->_cp->getPaddedInSize(), this->_cp->getInChannel());
		cudaThreadSynchronize();
		cudaCheckError();
	}else
		padded_x = x;

	//size表示一个正方形的边长，width，height表示矩阵的宽长
//padded_x->showValue("_x");	
	int num_box = pow(_cp->getBoxNumSize(), 2);

	dim3 blocks = dim3(_cp->getMinibatchSize(), _cp->getOutChannel()*num_box);
	dim3 threads = dim3(MAX_THREAD_SIZE, MAX_THREAD_SIZE);

	int box_out_size = MAX_THREAD_SIZE > _cp->getOutSize() \
					? _cp->getOutSize() : MAX_THREAD_SIZE;
	forward_convolution<<<blocks, threads>>>(\
			padded_x->getDevData(), this->_w->getDevData(), this->_bias->getDevData(), \
			this->_y->getDevData(), \
				_cp->getPaddedInSize(), _cp->getInChannel(), _cp->getOutSize(), \
				_cp->getFilterSize(), _cp->getOutChannel(), _cp->getStride(), \
				box_out_size, _cp->getBoxNumSize());  
	cudaThreadSynchronize();
	cudaCheckError();

//	this->_w->showValue("whk");
//	this->_y->showValue(this->_cp->getName() + "yh");
}

template <typename Dtype>
void ConvNet<Dtype>::computeDerivsOfPars(Matrix<Dtype>* x){
	x->reValue(500);
this->_dE_dy->reValue(1.0f);

	int num_box = pow(_cp->getBoxNumSize(), 2);

	dim3 blocks = dim3(_cp->getMinibatchSize(), _cp->getOutChannel()*_cp->getInChannel()*num_box);
	dim3 threads = dim3(MAX_THREAD_SIZE, MAX_THREAD_SIZE);

	int box_out_size = MAX_THREAD_SIZE > _cp->getOutSize() \
						? _cp->getOutSize() : MAX_THREAD_SIZE;

	int box_in_size = MAX_THREAD_SIZE > _cp->getOutSize() \
						  ? _cp->getInSize() : _cp->getBoxInSize();

	Dtype* p_dE_dw;
	if(MAX_THREAD_SIZE < _cp->getOutSize()){
		unranged_dE_dw->zeros();
		p_dE_dw = unranged_dE_dw->getDevData();
	}else{
		this->_dE_dw->zeros();
		p_dE_dw = this->_dE_dw->getDevData();
	}

	cudaStream_t s1, s2;
	cudaStreamCreate(&s1);
	cudaStreamCreate(&s2);

	compute_convolution_derivs<<<blocks, threads, sizeof(Dtype)*pow(_cp->getFilterSize(),2), s1>>>( \
			this->_dE_dy->getDevData(), x->getDevData(), p_dE_dw, \
			box_in_size, box_out_size, \
			_cp->getOutChannel(), _cp->getInChannel(), \
			_cp->getOutSize(), _cp->getFilterSize(), \
			_cp->getStride(), _cp->getBoxNumSize());	
	cudaThreadSynchronize();
	cudaCheckError();
		
	if(_cp->getOutSize() > MAX_THREAD_SIZE){
		int num_block = _cp->getInChannel()*_cp->getOutChannel();
		compact_dervis_w<<<num_block, threads>>>( \
				unranged_dE_dw->getDevData(), this->_dE_dw->getDevData(), \
				_cp->getFilterSize(), _cp->getBoxNumSize());
		cudaThreadSynchronize();
		cudaCheckError();
	}
	this->_dE_dw->showValue("dE_dw");

	

	//重排输出的导数来计算对b的导数
	int num_kernel = this->_cp->getMinibatchSize() * _conv_pixs \
				 * this->_cp->getOutChannel();
	int num_block = MAX_NUM_KERNEL < (num_kernel / MAX_NUM_THREAD + 1) ? MAX_NUM_KERNEL \
				: (num_kernel / MAX_NUM_THREAD + 1);
	reshape_dE_dy2<<<num_block, MAX_NUM_THREAD, 0, s2>>>(ranged_dE_dy2->getDevData(), \
			this->_dE_dy->getDevData(), num_kernel, this->_cp->getOutSize(), \
			this->_cp->getOutChannel());
	cudaThreadSynchronize();
	cudaCheckError();

	ranged_dE_dy2->sumCol(unrolled_dE_db_tmp);
//ranged_dE_dy2->showValue("dedy2");
//unrolled_dE_db_tmp->showValue("unrolled_dE_db_tmp");
	
	num_kernel = this->_cp->getMinibatchSize() * this->_cp->getOutChannel();
	num_block = MAX_NUM_KERNEL < (num_kernel / MAX_NUM_THREAD + 1) ? MAX_NUM_KERNEL \
				: (num_kernel / MAX_NUM_THREAD + 1);
	reshape_dE_db_tmp<<<num_block, MAX_NUM_THREAD, 0, s2>>>(dE_db_tmp->getDevData(), \
			unrolled_dE_db_tmp->getDevData(), num_kernel, this->_cp->getOutChannel());
	cudaThreadSynchronize();
	cudaCheckError();

//this->dE_db_tmp->showValue("dEdbtmp");
	dE_db_tmp->sumRow(this->_dE_db);
this->_dE_db->showValue(this->_cp->getName() + "dEdb");

	cudaStreamDestroy(s1);
	cudaStreamDestroy(s2);
}

template <typename Dtype>
void ConvNet<Dtype>::computeDerivsOfInput(Matrix<Dtype>* dE_dx){

	dE_dx->zeros();
	
//this->_dE_dy->reValue(16);
//this->_w->reValue(1.0f);

	int num_box = pow(_cp->getBoxNumSize(), 2);

	dim3 blocks = dim3(_cp->getMinibatchSize(), _cp->getInChannel() * num_box);
	dim3 threads = dim3(MAX_THREAD_SIZE, MAX_THREAD_SIZE);

	int box_out_size = MAX_THREAD_SIZE > _cp->getOutSize() \
						? _cp->getOutSize() : MAX_THREAD_SIZE;

	int box_in_size = MAX_THREAD_SIZE > _cp->getOutSize() \
						  ? _cp->getInSize() : _cp->getBoxInSize();

	Dtype* p_dE_dx;
	if(MAX_THREAD_SIZE < _cp->getOutSize() && _overlap_len > 0){
		unranged_dE_dx->zeros();
		p_dE_dx = unranged_dE_dx->getDevData();
	}else{
		dE_dx->zeros();
		p_dE_dx = dE_dx->getDevData();
	}

	backward_convolution<<<blocks, threads, sizeof(Dtype)*pow(box_in_size,2)>>>( \
			this->_dE_dy->getDevData(), this->_w->getDevData(), \
			p_dE_dx, box_in_size, box_out_size, \
			_cp->getOutChannel(), _cp->getInChannel(), \
			_cp->getOutSize(), _cp->getFilterSize(), \
			_cp->getStride(), _cp->getBoxNumSize());	
	
	cudaThreadSynchronize();
	cudaCheckError();


//	dE_dx->showValue(this->_cp->getName() + "dx");

}
