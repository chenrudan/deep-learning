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
void ConvNet<Dtype>::computeOutputs(Matrix<Dtype>* _x){

	int num_kernel;
	int num_block;
//	_x->reValue(32);
//	this->_w->reValue(1.0f);
//	this->_bias->reValue(2.0f);

	if(this->_cp->getPad() > 0){
		num_kernel = this->_cp->getMinibatchSize() * _padded_in_pixs \
					 * this->_cp->getInChannel();
		num_block = MAX_NUM_KERNEL < (num_kernel / MAX_NUM_THREAD + 1) \
					? MAX_NUM_KERNEL : (num_kernel / MAX_NUM_THREAD + 1);
		cudaMemset(padded_x->getDevData(), 0, sizeof(Dtype) * num_kernel);
		ori_to_padding<<<num_block, MAX_NUM_THREAD>>>(_x->getDevData(), \
				padded_x->getDevData(), num_kernel, this->_cp->getInSize(), \
				this->_cp->getPaddedInSize(), this->_cp->getInChannel());
	}else
		padded_x = _x;

//	padded_x->showValue("padding");
	num_kernel = this->_cp->getMinibatchSize() * _conv_pixs * _filt_pixs \
				 *this->_cp->getInChannel();
	num_block = num_kernel / MAX_NUM_THREAD + 1;
	cudaMemset(unrolled_x1->getDevData(), 0, sizeof(Dtype) * num_kernel);
	im2col_filt<<<num_block, MAX_NUM_THREAD>>>(padded_x->getDevData(), \
			unrolled_x1->getDevData(), num_kernel, this->_cp->getPaddedInSize(), \
			this->_cp->getInChannel(), this->_cp->getFilterSize(), \
			this->_cp->getOutSize(), this->_cp->getStride());

	unrolled_x1->rightMult(this->_w, 1, unranged_y, this->handle);

	unranged_y->addRowVector(this->_bias);

	num_kernel = this->_cp->getMinibatchSize() * _conv_pixs \
				 * this->_cp->getOutChannel();
	num_block = MAX_NUM_KERNEL < (num_kernel / MAX_NUM_THREAD + 1) ? MAX_NUM_KERNEL \
				: (num_kernel / MAX_NUM_THREAD + 1);
	reshape_y<<<num_block, MAX_NUM_THREAD>>>(unranged_y->getDevData(), \
			this->_y->getDevData(), num_kernel, this->_cp->getOutSize(), \
			this->_cp->getOutChannel());
	//unrolled_x1->showValue("data");
//	this->_w->showValue("whk");
//	this->_y->showValue("yh");
}

template <typename Dtype>
void ConvNet<Dtype>::computeDerivsOfPars(Matrix<Dtype>* x){
	
//this->_dE_dy->showValue("dedy");
//	x->reValue(32);
//	x->showValue("x");
//	padded_x->showValue("pad");

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

//	unrolled_x2->showValue("x2");
//	this->_dE_dy->reValue(32);
//unrolled_x2->reValue(1);

	num_kernel = this->_cp->getMinibatchSize() * _conv_pixs \
				 * this->_cp->getOutChannel();
	num_block = MAX_NUM_KERNEL < (num_kernel / MAX_NUM_THREAD + 1) ? MAX_NUM_KERNEL \
				: (num_kernel / MAX_NUM_THREAD + 1);
	reshape_dE_dy<<<num_block, MAX_NUM_THREAD>>>(ranged_dE_dy->getDevData(), \
			this->_dE_dy->getDevData(), num_kernel, this->_cp->getOutSize(), \
			this->_cp->getOutChannel());

	unrolled_x2->rightMult(ranged_dE_dy, 1, this->_dE_dw, this->handle);

//ranged_dE_dy->showValue("dedxdh");	
//this->_dE_dy->showValue("dedy");
//this->_dE_dw->showValue("dedwhk");

	//重排输出的导数来计算对b的导数
	num_kernel = this->_cp->getMinibatchSize() * _conv_pixs \
				 * this->_cp->getOutChannel();
	num_block = MAX_NUM_KERNEL < (num_kernel / MAX_NUM_THREAD + 1) ? MAX_NUM_KERNEL \
				: (num_kernel / MAX_NUM_THREAD + 1);
	reshape_dE_dy2<<<num_block, MAX_NUM_THREAD>>>(ranged_dE_dy2->getDevData(), \
			this->_dE_dy->getDevData(), num_kernel, this->_cp->getOutSize(), \
			this->_cp->getOutChannel());
	cudaThreadSynchronize();
	ranged_dE_dy2->sumCol(unrolled_dE_db_tmp);
	
	num_kernel = this->_cp->getMinibatchSize() * this->_cp->getOutChannel();
	num_block = MAX_NUM_KERNEL < (num_kernel / MAX_NUM_THREAD + 1) ? MAX_NUM_KERNEL \
				: (num_kernel / MAX_NUM_THREAD + 1);
	reshape_dE_db_tmp<<<num_block, MAX_NUM_THREAD>>>(dE_db_tmp->getDevData(), \
			unrolled_dE_db_tmp->getDevData(), num_kernel, this->_cp->getOutChannel());

	dE_db_tmp->sumRow(this->_dE_db);
}

template <typename Dtype>
void ConvNet<Dtype>::computeDerivsOfInput(Matrix<Dtype>* dE_dx){
	
	int num_kernel = this->_cp->getMinibatchSize() * _padded_in_pixs \
					 * _filt_pixs * this->_cp->getOutChannel();
	int	num_block = MAX_NUM_KERNEL < (num_kernel / MAX_NUM_THREAD + 1) \
					? MAX_NUM_KERNEL : (num_kernel / MAX_NUM_THREAD + 1);

//this->_dE_dy->reValue(16);

	cudaMemset(unrolled_conv->getDevData(), 0, sizeof(Dtype) * num_kernel);
	im2col_img<<<num_block, MAX_NUM_THREAD>>>(this->_dE_dy->getDevData(), \
			unrolled_conv->getDevData(), num_kernel, \
			this->_cp->getPaddedInSize(), this->_cp->getOutChannel(), \
			this->_cp->getInChannel(), this->_cp->getFilterSize(), \
			this->_cp->getOutSize(), this->_cp->getStride());
	cudaThreadSynchronize();
//this->_w->reValue(1.0f);
	num_kernel = this->_cp->getOutChannel() * _filt_pixs * this->_cp->getInChannel();
	num_block = num_kernel / MAX_NUM_THREAD + 1;
	reshape_w<<<num_block, MAX_NUM_THREAD>>>(ranged_w->getDevData(), \
			this->_w->getDevData(), num_kernel, this->_cp->getFilterSize(), \
			this->_cp->getOutChannel(), this->_cp->getInChannel());
	cudaThreadSynchronize();
	
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
	
//	this->_w->showValue("whk");
//	unrolled_conv->showValue("unrolledconv");
//	rangedthis->_w->showValue("rangWhk");
//		unranged_in->showValue("unrangIN");
//	dE_dx->showValue("dx");

}






























