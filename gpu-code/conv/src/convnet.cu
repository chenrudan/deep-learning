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
	
	delete unrolled_x1;
	delete unranged_y;
	delete unrolled_x2;
	delete ranged_dE_dy;
	delete ranged_dE_dy2;
	delete unrolled_dE_db_tmp;
	delete dE_db_tmp;
	delete unrolled_conv;
	delete ranged_w;
	delete unranged_in;
	if(this->_cp->getPad() > 0)
		delete padded_x;

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
	
	unrolled_x1 			= new Matrix<Dtype>(this->_cp->getMinibatchSize() \
									* _conv_pixs, \
									_filt_pixs * this->_cp->getInChannel());   
	///>变换排列方式用来做矩阵乘法计算卷积
	unranged_y 				= new Matrix<Dtype>(this->_cp->getMinibatchSize() \
									* _conv_pixs, \
									this->_cp->getOutChannel());

	unrolled_x2 			= new Matrix<Dtype>(_filt_pixs \
									* this->_cp->getInChannel(), \
									this->_cp->getMinibatchSize() * _conv_pixs);
	ranged_dE_dy 			= new Matrix<Dtype>(this->_cp->getMinibatchSize() \
									* _conv_pixs, \
									this->_cp->getOutChannel());
	ranged_dE_dy2 			= new Matrix<Dtype>(this->_cp->getMinibatchSize() \
									* this->_cp->getOutChannel(), \
									_conv_pixs);
	unrolled_dE_db_tmp 		= new Matrix<Dtype>(this->_cp->getMinibatchSize() \
									* this->_cp->getOutChannel(), 1);
	dE_db_tmp 				= new Matrix<Dtype>(this->_cp->getMinibatchSize(), \
									this->_cp->getOutChannel());

	unrolled_conv 			= new Matrix<Dtype>(this->_cp->getMinibatchSize() \
 									* _padded_in_pixs, \
									_filt_pixs * this->_cp->getOutChannel());
	ranged_w 				= new Matrix<Dtype>(this->_cp->getOutChannel() \
									* _filt_pixs, this->_cp->getInChannel());
	unranged_in 			= new Matrix<Dtype>(this->_cp->getMinibatchSize() * \
									_padded_in_pixs, this->_cp->getInChannel());

	this->_w_inc->zeros();
	this->_bias_inc->zeros();
}

template <typename Dtype>
void ConvNet<Dtype>::computeOutput(Matrix<Dtype>* x){

	this->_y->zeros();

	int num_kernel;
	int num_block;
//	x->reValue(32);
//	this->_w->reValue(1.0f);
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

//padded_x->showValue("_x");	
	//size表示一个正方形的边长，width，height表示矩阵的宽长
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

	this->_dE_dw->zeros();
	this->_dE_db->zeros();
//this->_dE_dy->showValue("dedy");
//	x->showValue("x");
//	padded_x->reValue(100);

	int num_kernel = this->_cp->getMinibatchSize() * _conv_pixs * _filt_pixs \
					 * this->_cp->getInChannel();
	int num_block = MAX_NUM_KERNEL < (num_kernel / MAX_NUM_THREAD + 1) \
					? MAX_NUM_KERNEL : (num_kernel / MAX_NUM_THREAD + 1);
	cudaMemset(unrolled_x2->getDevData(), 0, sizeof(Dtype) * num_kernel);
	im2col_conv<<<num_block, MAX_NUM_THREAD>>>(padded_x->getDevData(), \
			unrolled_x2->getDevData(), num_kernel, \
			this->_cp->getMinibatchSize(), this->_cp->getPaddedInSize(), \
			this->_cp->getInChannel(), this->_cp->getFilterSize(), \
			this->_cp->getOutSize(), this->_cp->getStride());	
	cudaThreadSynchronize();
	cudaCheckError();

//	unrolled_x2->showValue("x2");
//	this->_dE_dy->reValue(1.0f);
	
	num_kernel = this->_cp->getMinibatchSize() * _conv_pixs \
				 * this->_cp->getOutChannel();
	num_block = MAX_NUM_KERNEL < (num_kernel / MAX_NUM_THREAD + 1) ? MAX_NUM_KERNEL \
				: (num_kernel / MAX_NUM_THREAD + 1);
	reshape_dE_dy<<<num_block, MAX_NUM_THREAD>>>(ranged_dE_dy->getDevData(), \
			this->_dE_dy->getDevData(), num_kernel, this->_cp->getOutSize(), \
			this->_cp->getOutChannel());
	cudaThreadSynchronize();
	cudaCheckError();

	unrolled_x2->rightMult(ranged_dE_dy, 1, this->_dE_dw, this->handle);

//ranged_dE_dy->showValue("dedxdh");
/*	
if(this->_cp->getName() == "conv1_layer"){
//	padded_x->showValue(this->_cp->getName() + "padding");
	this->_dE_dw->showValue(this->_cp->getName() + "dedw");
//	this->_dE_dy->showValue(this->_cp->getName() + "dedy");
	this->_w->showValue(this->_cp->getName() + "w");
	this->_y->showValue(this->_cp->getName() + "yh")
}*/

	//重排输出的导数来计算对b的导数
	num_kernel = this->_cp->getMinibatchSize() * _conv_pixs \
				 * this->_cp->getOutChannel();
	num_block = MAX_NUM_KERNEL < (num_kernel / MAX_NUM_THREAD + 1) ? MAX_NUM_KERNEL \
				: (num_kernel / MAX_NUM_THREAD + 1);
	reshape_dE_dy2<<<num_block, MAX_NUM_THREAD>>>(ranged_dE_dy2->getDevData(), \
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
	reshape_dE_db_tmp<<<num_block, MAX_NUM_THREAD>>>(dE_db_tmp->getDevData(), \
			unrolled_dE_db_tmp->getDevData(), num_kernel, this->_cp->getOutChannel());
	cudaThreadSynchronize();
	cudaCheckError();

//this->dE_db_tmp->showValue("dEdbtmp");
	dE_db_tmp->sumRow(this->_dE_db);
//this->_dE_db->showValue(this->_cp->getName() + "dEdb");
}

template <typename Dtype>
void ConvNet<Dtype>::computeDerivsOfInput(Matrix<Dtype>* dE_dx){

	dE_dx->zeros();
	
//this->_dE_dy->reValue(44);

	int num_box = pow(_lcp->getBoxNumSize(), 2);

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

	


	/*

	cudaMemset(unrolled_conv->getDevData(), 0, sizeof(Dtype) * num_kernel);
	im2col_img<<<num_block, MAX_NUM_THREAD>>>(this->_dE_dy->getDevData(), \
			unrolled_conv->getDevData(), num_kernel, \
			this->_cp->getPaddedInSize(), this->_cp->getOutChannel(), \
			this->_cp->getInChannel(), this->_cp->getFilterSize(), \
			this->_cp->getOutSize(), this->_cp->getStride());
	cudaThreadSynchronize();
	cudaCheckError();
//this->_w->reValue(1.0f);
	num_kernel = this->_cp->getOutChannel() * _filt_pixs * this->_cp->getInChannel();
	num_block = num_kernel / MAX_NUM_THREAD + 1;
	reshape_w<<<num_block, MAX_NUM_THREAD>>>(ranged_w->getDevData(), \
			this->_w->getDevData(), num_kernel, this->_cp->getFilterSize(), \
			this->_cp->getOutChannel(), this->_cp->getInChannel());
	cudaThreadSynchronize();
	cudaCheckError();
	
	unrolled_conv->rightMult(ranged_w, 1, unranged_in, this->handle);
	
	num_kernel = this->_cp->getMinibatchSize() * _padded_in_pixs \
				 * this->_cp->getInChannel();
	num_block = MAX_NUM_KERNEL < (num_kernel / MAX_NUM_THREAD + 1) ? MAX_NUM_KERNEL \
				: (num_kernel / MAX_NUM_THREAD + 1);

//unranged_in->reValue(20*16);

	reshape_In<<<num_block, MAX_NUM_THREAD>>>(dE_dx->getDevData(), \
			unranged_in->getDevData(), num_kernel, \
			this->_cp->getInSize(), this->_cp->getPaddedInSize(), \
			this->_cp->getInChannel());
	cudaThreadSynchronize();
	cudaCheckError();
	*/
	
//	this->_w->showValue("whk");
//	unrolled_conv->showValue("unrolledconv");
//	rangedthis->_w->showValue("rangWhk");
//		unranged_in->showValue("unrangIN");
//	dE_dx->showValue(this->_cp->getName() + "dx");

}






























