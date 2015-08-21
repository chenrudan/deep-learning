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
	this->_in_pixs		= pow(this->_cp->getInSize(), 2);
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

	delete unfold_x;
	delete dE_db_tmp;
	if(this->_cp->getPad() > 0)
		delete padded_x;
	if(_cp->getOutSize() > MAX_THREAD_SIZE && _overlap_len > 0)	
		delete unranged_dE_dx;
	if(_cp->getOutSize() > MAX_THREAD_SIZE){
		delete unranged_dE_dw;
		delete unfold_dE_db_tmp;
	}

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
	unfold_x 		= new Matrix<Dtype>(this->_cp->getMinibatchSize(), \
									this->_cp->getInChannel() * _padded_in_pixs);
	
	_overlap_len = _cp->getFilterSize() - _cp->getStride();
	if(_cp->getOutSize() > MAX_THREAD_SIZE && _overlap_len > 0){	
		unranged_dE_dx = new Matrix<Dtype>(_cp->getMinibatchSize(), \
				pow(_cp->getBoxInSize() * _cp->getBoxNumSize(), 2) \
				* _cp->getOutChannel());
	}
	unranged_dE_dw = new Matrix<Dtype>(_cp->getMinibatchSize(), \
			pow(_cp->getFilterSize(),2)*_cp->getInChannel() \
			*pow(_cp->getBoxNumSize(),2)*_cp->getOutChannel());
	if(_cp->getOutSize() > MAX_THREAD_SIZE){	
		unfold_dE_db_tmp		 = new Matrix<Dtype>(this->_cp->getMinibatchSize(), \
									this->_cp->getOutChannel()*pow(_cp->getBoxNumSize(), 2));
	}

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
		num_kernel = this->_cp->getMinibatchSize() * _in_pixs \
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
	padded_x->reValue(1.0f);
this->_dE_dy->reValue(1.0f);

	int num_box = pow(_cp->getBoxNumSize(), 2);

	dim3 blocks = dim3(_cp->getMinibatchSize(), _cp->getOutChannel()*_cp->getInChannel()*num_box);
	dim3 threads = dim3(MAX_THREAD_SIZE, MAX_THREAD_SIZE);

	int box_out_size = MAX_THREAD_SIZE > _cp->getOutSize() \
						? _cp->getOutSize() : MAX_THREAD_SIZE;

	int box_in_size = MAX_THREAD_SIZE > _cp->getOutSize() \
						  ? _cp->getPaddedInSize() : _cp->getBoxInSize();

	unranged_dE_dw->zeros();

	cudaStream_t s1, s2;
	cudaStreamCreate(&s1);
	cudaStreamCreate(&s2);

	compute_convolution_derivs<<<blocks, threads, \
		sizeof(Dtype)*pow(_cp->getFilterSize(),2), s1>>>( \
			this->_dE_dy->getDevData(), padded_x->getDevData(), \
			unranged_dE_dw->getDevData(), \
			box_in_size, box_out_size, \
			_cp->getOutChannel(), _cp->getInChannel(), \
			_cp->getOutSize(), _cp->getFilterSize(), \
			_cp->getStride(), _cp->getBoxNumSize());	
	cudaThreadSynchronize();
	cudaCheckError();
	
	blocks = (1, _cp->getInChannel()*_cp->getOutChannel());
	compact_dervis_w<<<blocks, threads>>>( \
				unranged_dE_dw->getDevData(), this->_dE_dw->getDevData(), \
				_cp->getFilterSize(), _cp->getBoxNumSize(), _cp->getMinibatchSize(), \
				_cp->getInChannel(), _cp->getOutChannel());
		cudaThreadSynchronize();
		cudaCheckError();
	
	if(_cp->getName() == "conv1"){
//		unranged_dE_dw->showValue("unranged_dE_dw");
		this->_dE_dw->showValue("dE_dw");
	}

	Dtype *dE_db_multi_channel;
	if(_cp->getOutSize() > MAX_THREAD_SIZE){
		unfold_dE_db_tmp->zeros();
		dE_db_multi_channel = unfold_dE_db_tmp->getDevData();
	}else{
		dE_db_tmp->zeros();
		dE_db_multi_channel = dE_db_tmp->getDevData();

	}

	blocks = dim3(_cp->getMinibatchSize(), _cp->getOutChannel()*num_box);
	compute_derivs_of_bias<<<blocks, threads, sizeof(Dtype), \
			s2>>>(this->_dE_dy->getDevData(), dE_db_multi_channel, \
					_cp->getOutSize(), _cp->getOutChannel(), \
					box_out_size, _cp->getBoxNumSize());
	cudaThreadSynchronize();
	cudaCheckError();

	if(_cp->getOutSize() > MAX_THREAD_SIZE){
		blocks = dim3(_cp->getMinibatchSize(), _cp->getOutChannel());
		compute_derivs_of_bias<<<blocks, threads, sizeof(Dtype), \
				s2>>>(unfold_dE_db_tmp->getDevData(), dE_db_tmp->getDevData(), \
						_cp->getBoxNumSize(), _cp->getOutChannel(), 1, 1);
	}
//dE_db_tmp->showValue(this->_cp->getName() + "dEdbtmp");

	dE_db_tmp->sumRow(this->_dE_db);

//this->_dE_db->showValue(this->_cp->getName() + "dEdb");

	cudaStreamDestroy(s1);
	cudaStreamDestroy(s2);
}

template <typename Dtype>
void ConvNet<Dtype>::computeDerivsOfInput(Matrix<Dtype>* dE_dx){

	
//this->_dE_dy->reValue(31);
//this->_w->reValue(1.0f);

	int num_box = pow(_cp->getBoxNumSize(), 2);

	dim3 blocks = dim3(_cp->getMinibatchSize(), _cp->getInChannel() * num_box);
	dim3 threads = dim3(MAX_THREAD_SIZE, MAX_THREAD_SIZE);

	int box_out_size = MAX_THREAD_SIZE > _cp->getOutSize() \
						? _cp->getOutSize() : MAX_THREAD_SIZE;

	int box_in_size = MAX_THREAD_SIZE > _cp->getOutSize() \
						  ? _cp->getPaddedInSize() : _cp->getBoxInSize();

	Dtype* p_dE_dx;
	if(MAX_THREAD_SIZE < _cp->getOutSize() && _overlap_len > 0){
		unranged_dE_dx->zeros();
		p_dE_dx = unranged_dE_dx->getDevData();
	}else if(_cp->getPad() > 0){
		unfold_x->zeros();
		p_dE_dx = unfold_x->getDevData();
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
//unfold_x->showValue(this->_cp->getName() + "dx");

	if(_cp->getOutSize() > MAX_THREAD_SIZE && _overlap_len > 0){
		
		if(this->_cp->getPad() > 0){
			unfold_x->zeros();
			p_dE_dx = unfold_x->getDevData();
		}else{
			dE_dx->zeros();
			p_dE_dx = dE_dx->getDevData();
		}
		
		compactOverlap<<<_cp->getMinibatchSize(), _cp->getInChannel()>>>( \
				unranged_dE_dx->getDevData(), p_dE_dx, _cp->getPaddedInSize(), \
				_cp->getInChannel(),  _overlap_len, \
				_cp->getBoxInSize(), _cp->getBoxNumSize());
		cudaThreadSynchronize();
		cudaCheckError();
//unranged_dE_dx->showValue("unrangeddEdx");
	}
	if(this->_cp->getPad() > 0){
		int num_kernel = this->_cp->getMinibatchSize() * _in_pixs \
					 * this->_cp->getInChannel();
		int num_block = MAX_NUM_KERNEL < (num_kernel / MAX_NUM_THREAD + 1) \
					? MAX_NUM_KERNEL : (num_kernel / MAX_NUM_THREAD + 1);
		pad_to_ori<<<num_block, MAX_NUM_THREAD>>>(dE_dx->getDevData(), \
				p_dE_dx, num_kernel, this->_cp->getInSize(), \
				this->_cp->getPaddedInSize(), this->_cp->getInChannel());
		cudaThreadSynchronize();
		cudaCheckError();

//dE_dx->showValue(this->_cp->getName() + "dx");
			
	}
	
}
