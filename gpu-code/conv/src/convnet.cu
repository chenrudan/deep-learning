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
	this->_filt_pixs			= this->_cp->getFilterHeight()*cp-->getFilterWidth();
	this->_conv_pixs			= this->_cp->getOutHeight()*_cp->getOutWidth();
	this->_padded_in_pixs		= this->_cp->getPaddedInHeight()*cp->getPaddedInWidth();
	this->_in_pixs				= this->_cp->getInHeight()*_cp->getInWidth();
	this->_box_in_pixs			= this->_cp->getBoxInHeight()*_cp->getBoxInWidth();
	cublasCreate(&this->handle);
	
	_num_box = _cp->getBoxNumHeight()*_cp->getBoxNumWidth();
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
	if(_cp->getPadHeight() > 0 || _cp->getPadWidth() > 0){
		delete padded_x;
	if((_cp->getOutHeight() > MAX_THREAD_SIZE \
				|| _cp->getOutWidth() > MAX_THREAD_SIZE) \
			&& (_cp->getOverlapHeight() > 0 || _cp->getOverlapWidth() > 0))	
		delete unranged_dE_dx;
	if(_cp->getOutHeight() > MAX_THREAD_SIZE || _cp->getOutWidth() > MAX_THREAD_SIZE){
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

	if(_cp->getPadHeight() > 0 || _cp->getPadWidth() > 0){
		this->padded_x 		= new Matrix<Dtype>(this->_cp->getMinibatchSize(), \
									this->_cp->getInChannel() * _padded_in_pixs);
	unfold_x 		= new Matrix<Dtype>(this->_cp->getMinibatchSize(), \
									this->_cp->getInChannel() * _padded_in_pixs);
	
	_cp->getOverlapHeight() = _cp->getFilterHeight() - _cp->getStrideHeight();
	_cp->getOverlapWidth() = _cp->getFilterWidth() - _cp->getStrideWidth();

	if((_cp->getOutHeight() > MAX_THREAD_SIZE \
				|| _cp->getOutWidth() > MAX_THREAD_SIZE) \
			&& (_cp->getOverlapHeight() > 0 || _cp->getOverlapWidth() > 0)){
		unranged_dE_dx = new Matrix<Dtype>(_cp->getMinibatchSize(), \
				_box_in_pixs*_num_box*_cp->getOutChannel());
	}
	unranged_dE_dw = new Matrix<Dtype>(_cp->getMinibatchSize(), \
			_filt_pixs*_cp->getInChannel() \
			_num_box*_cp->getOutChannel());

	if(_cp->getOutHeight() > MAX_THREAD_SIZE \
				|| _cp->getOutWidth() > MAX_THREAD_SIZE) {
		unfold_dE_db_tmp		 = new Matrix<Dtype>(this->_cp->getMinibatchSize(), \
									this->_cp->getOutChannel()*_num_box;
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
//	this->_w->reValue(_filt_pixs, true);
//	this->_bias->reValue(2.0f);

	if(_cp->getPadHeight() > 0 || _cp->getPadWidth() > 0){
		num_kernel = this->_cp->getMinibatchSize() * _in_pixs \
					 * this->_cp->getInChannel();
		num_block = MAX_NUM_KERNEL < (num_kernel / MAX_NUM_THREAD + 1) \
					? MAX_NUM_KERNEL : (num_kernel / MAX_NUM_THREAD + 1);
		padded_x->zeros();
		ori_to_padding<<<num_block, MAX_NUM_THREAD>>>(x->getDevData(), \
				padded_x->getDevData(), num_kernel, this->_cp->getInHeight(), \
				_cp->getInWidth(), _cp->getPaddedInHeight(), \
				_cp->getPaddedInWidth(), _cp->getInChannel());
		cudaDeviceSynchronize();
		cudaCheckError();
	}else
		padded_x = x;

	//size表示一个正方形的边长，width，height表示矩阵的宽长
//if(_cp->getName() == "conv3")
//padded_x->showValue("_x");	

	dim3 blocks = dim3(_cp->getMinibatchSize(), _cp->getOutChannel()*_num_box);
	dim3 threads = dim3(_cp->getThreadHeight(), _cp->getThreadWidth());

	forward_convolution<<<blocks, threads>>>(\
			padded_x->getDevData(), this->_w->getDevData(), \
			this->_bias->getDevData(), this->_y->getDevData(), \
			_cp->getPaddedInHeight(), _cp->getPaddedInWidth(), \
			_cp->getInChannel(), _cp->getOutHeight(), \
			_cp->getOutWidth(), _cp->getFilterHeight(), \
			_cp->getFilterWidth(), _cp->getOutChannel(), \
			_cp->getStrideHeight(), _cp->getStrideWidth(), \
			_cp->getBoxNumHeight(), _cp->getBoxNumWidth());  
	cudaDeviceSynchronize();
	cudaCheckError();


//	this->_w->showValue("whk");
//	this->_y->showValue(this->_cp->getName() + "yh");
}

template <typename Dtype>
void ConvNet<Dtype>::computeDerivsOfPars(Matrix<Dtype>* x){

//	padded_x->reValue(1.0f);
//this->_dE_dy->reValue(_cp->getOutSize()*_cp->getOutSize(),true);

//	clock_t t;
//	t = clock();


	dim3 blocks = dim3(_cp->getMinibatchSize(), \
			_cp->getOutChannel()*_cp->getInChannel()*_num_box \
			*_cp->getFilterHeight()*_cp->getFilterWidth());

	dim3 threads = dim3(_cp->getThreadHeight(), _cp->getThreadWidth());

	unranged_dE_dw->zeros();

	Dtype *dE_db_multi_channel;
	if(_cp->getOutHeight() > MAX_THREAD_SIZE \
				|| _cp->getOutWidth() > MAX_THREAD_SIZE) {
		unfold_dE_db_tmp->zeros();
		dE_db_multi_channel = unfold_dE_db_tmp->getDevData();
	}else{
		dE_db_tmp->zeros();
		dE_db_multi_channel = dE_db_tmp->getDevData();
	}

	cudaStream_t s1, s2;
	cudaStreamCreate(&s1);
	cudaStreamCreate(&s2);

	//每个线程只计算一个点，先将32*32输出块计算对应值存入共享变量，再用reduce
	compute_convolution_derivs<<<blocks, threads, \
		sizeof(Dtype)*_cp->getBoxOutHeight()*_cp->getBoxOutWidth(), s1>>>( \
			this->_dE_dy->getDevData(), padded_x->getDevData(), \
			unranged_dE_dw->getDevData(), \
			_cp->getBoxOutHeight(), _cp->getBoxOutWidth(), \
			_cp->getOutChannel(), _cp->getInChannel(), \
			_cp->getPaddedInHeight(), _cp->getPaddedInWidth(), \
			_cp->getOutHeight(), _cp->getOutWidth(), \
			_cp->getFilterHeight(), _cp->getFilterWidth(), \
			_cp->getStrideHeight(), _cp->getStrideWidth(), \
			_cp->getBoxNumHeight(), _cp->getBoxNumWidth());  
	
	blocks = dim3(_cp->getMinibatchSize(), _cp->getOutChannel()*_num_box);
	//从100*out_size*out_size*out_size*out_channel先生成100*out_channel*_num_box
	compute_derivs_of_bias<<<blocks, threads, \
		sizeof(Dtype)*_cp->getBoxOutHeight()*cp->getBoxOutWidth(), s2>>>( \
			this->_dE_dy->getDevData(), dE_db_multi_channel, \
			_cp->getOutHeight(), _cp->getOutWidth(), \
			_cp->getOutChannel(), \
			_cp->getBoxOutHeight(), _cp->getBoxOutWidth(), \
			_cp->getBoxNumHeight(), _cp->getBoxNumWidth());  

	cudaDeviceSynchronize();
	cudaCheckError();
	
//	t = clock() - t;
//	cout << _cp->getName() << " dervis w  convolution: "<< ((float)t/CLOCKS_PER_SEC) << "s.\n";
//	t = clock();

	blocks = dim3(1, _cp->getInChannel()*_cp->getOutChannel());
	compact_dervis_w<<<blocks, threads, 0, s1>>>( \
			unranged_dE_dw->getDevData(), this->_dE_dw->getDevData(), \
			_cp->getFilterHeight(), _cp->getFilterWidth(), \
			_cp->getBoxNumHeight(), _cp->getBoxNumWidth() \
			_cp->getMinibatchSize(), _cp->getInChannel(), _cp->getOutChannel());
	
	if(_cp->getOutHeight() > MAX_THREAD_SIZE \
				|| _cp->getOutWidth() > MAX_THREAD_SIZE) {
		blocks = dim3(_cp->getMinibatchSize(), _cp->getOutChannel());
		compute_derivs_of_bias<<<blocks, threads, sizeof(Dtype)*_num_box, s2>>>( \
				unfold_dE_db_tmp->getDevData(), dE_db_tmp->getDevData(), \
						_cp->getBoxNumHeight(), _cp->getBoxNumWidth(), \
						_cp->getOutChannel(), _cp->getBoxNumHeight(), \
						_cp->getBoxNumWidth(), 1, 1);
	}
	cudaDeviceSynchronize();
	cudaCheckError();
	
	dE_db_tmp->sumRow(this->_dE_db);

//	t = clock() - t;
//	cout << _cp->getName() << " compact w  convolution: "<< ((float)t/CLOCKS_PER_SEC) << "s.\n";

//	if(_cp->getName() != "conv3"){
//		this->_dE_dw->showValue("dE_dw");
//	unfold_dE_db_tmp->showValue(this->_cp->getName() + "dEdb");
//	}

//unranged_dE_dw->showValue(_cp->getName()+"unranged_dE_dw");
//this->_dE_dw->showValue(_cp->getName() + "dE_dw");
//this->_dE_db->showValue(this->_cp->getName() + "dEdb");

	cudaStreamDestroy(s1);
	cudaStreamDestroy(s2);

}

template <typename Dtype>
void ConvNet<Dtype>::computeDerivsOfInput(Matrix<Dtype>* dE_dx){

	
//	clock_t t;
//	t = clock();
//this->_dE_dy->reValue(1.0f);
//this->_w->reValue(_filt_pixs, true);
//this->_w->showValue("w");


	dim3 blocks = dim3(_cp->getMinibatchSize(), _cp->getInChannel() * _num_box);
	dim3 threads = dim3(_cp->getThreadHeight(), _cp->getThreadWidth());

	int box_in_height = MAX_THREAD_SIZE > _cp->getOutHeight() \
						  ? _cp->getPaddedInHeight() : _cp->getBoxInHeight();
	int box_in_width = MAX_THREAD_SIZE > _cp->getOutWidth() \
						  ? _cp->getPaddedInWidth() : _cp->getBoxInWidth();

	Dtype* p_dE_dx;
	if((_cp->getOutHeight() > MAX_THREAD_SIZE \
				|| _cp->getOutWidth() > MAX_THREAD_SIZE) \
			&& (_cp->getOverlapHeight() > 0 || _cp->getOverlapWidth() > 0)){
		unranged_dE_dx->zeros();
		p_dE_dx = unranged_dE_dx->getDevData();
	}else if(_cp->getPadHeight() > 0 || _cp->getPadWidth() > 0){
		unfold_x->zeros();
		p_dE_dx = unfold_x->getDevData();
	}else{
		dE_dx->zeros();
		p_dE_dx = dE_dx->getDevData();
	}

	backward_convolution<<<blocks, threads, \
		sizeof(Dtype)*box_in_height*box_in_width>>>( \
			this->_dE_dy->getDevData(), this->_w->getDevData(), \
			p_dE_dx, box_in_height, box_in_width, \
			_cp->getBoxOutHeight(), _cp->getBoxOutWidth(), \
			_cp->getOutChannel(), _cp->getInChannel(), \
			_cp->getOutHeight(), _cp->getOutWidth(), \
			_cp->getFilterHeight(), _cp->getFilterWidth(), \
			_cp->getStrideHeight(), _cp->getStrideWidth(), \
			_cp->getBoxNumHeight(), _cp->getBoxNumWidth());  
	cudaDeviceSynchronize();
	cudaCheckError();
//unfold_x->showValue(this->_cp->getName() + "dx");

	if((_cp->getOutHeight() > MAX_THREAD_SIZE \
				|| _cp->getOutWidth() > MAX_THREAD_SIZE) \
			&& (_cp->getOverlapHeight() > 0 || _cp->getOverlapWidth() > 0)){
		
		if(_cp->getPadHeight() > 0 || _cp->getPadWidth() > 0){
			unfold_x->zeros();
			p_dE_dx = unfold_x->getDevData();
		}else{
			dE_dx->zeros();
			p_dE_dx = dE_dx->getDevData();
		}
		
		compactOverlap<<<_cp->getMinibatchSize(), _cp->getInChannel()>>>( \
				unranged_dE_dx->getDevData(), p_dE_dx, \
				_cp->getPaddedInHeight(), _cp->getPaddedInWidth(), \
				_cp->getInChannel(),  _cp->getOverlapHeight(), _cp->getOverlapWidth(), \
				_cp->getBoxInHeight(), _cp->getBoxInWidth(), \
				_cp->getBoxNumHeight(), _cp->getBoxNumWidth());
		cudaDeviceSynchronize();
		cudaCheckError();
//unranged_dE_dx->showValue("unrangeddEdx");
	}
	if(_cp->getPadHeight() > 0 || _cp->getPadWidth() > 0){
		int num_kernel = this->_cp->getMinibatchSize() * _in_pixs \
					 * this->_cp->getInChannel();
		int num_block = MAX_NUM_KERNEL < (num_kernel / MAX_NUM_THREAD + 1) \
					? MAX_NUM_KERNEL : (num_kernel / MAX_NUM_THREAD + 1);
		pad_to_ori<<<num_block, MAX_NUM_THREAD>>>(dE_dx->getDevData(), \
				p_dE_dx, num_kernel, _cp->getInHeight(), _cp->getInWidth(), \
				_cp->getPaddedInHeight(), _cp->getPaddedInWidth(), \
				_cp->getInChannel());
		cudaDeviceSynchronize();
		cudaCheckError();

//unfold_x->showValue(this->_cp->getName() + "unfolddx");
//dE_dx->showValue(this->_cp->getName() + "dx");
			
	}

//	t = clock() - t;
//	cout << _cp->getName() << " backward convolution: "<< ((float)t/CLOCKS_PER_SEC) << "s.\n";
}
