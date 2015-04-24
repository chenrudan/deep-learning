/*
 * filename: convnet.cu
 */
//#include <cutil_inline.h>
#include <time.h>

#include "convnet.cuh"
#include "convnet_kernel.cuh"

using namespace std;


ConvNet::ConvNet(pars* netWork){

	this->_filter_channel           = netWork->filter_channel;

	this->_w_lr             		= netWork->w_lr;
	//hidden bias的learning rate
	this->_b_lr            			= netWork->b_lr;
	//上一次更新的参数控制增长趋势
	this->_momentum                 = netWork->momentum;
	//hidden原值的参数
	this->_weight_decay             = netWork->weight_decay;
	//out原值的参数
	this->_minibatch_size         	= netWork->minibatch_size;
	this->_in_size				 	= netWork->in_size;
	this->_filter_size			 	= netWork->filter_size;
	this->_stride              		= netWork->stride;
	this->_out_size		 			= (_in_size - _filter_size) / _stride + 1;
	this->_numTrain				 	= netWork->trainNum;
	this->_numValid				 	= netWork->validNum;
	this->_in_channel			 	= netWork->in_channel;
	this->_lr_down_scale			= netWork->lr_down_scale;
	this->_filt_pixs				= _filter_size * _filter_size;
	this->_conv_pixs				= _out_size * _out_size;
	cublasCreate(&handle);
}
ConvNet::~ConvNet() {

	delete _w;
	delete _w_inc;
	delete _bias;
	delete _bias_inc;

	delete _y; 
	delete _dE_dy;
	delete _dE_dw;
	delete _dE_db;

	delete unrolled_x1;
	delete unranged_y;
	delete w_T;
	delete bias_T;
	delete unrolled_x2;
	delete ranged_dE_dx;
	delete dE_dw_T;
	delete dE_db_tmp;
	delete unrolled_conv;
	delete ranged_w;
	delete unranged_in;

	cublasDestroy(handle);
}

void ConvNet::initCuda() {
	//cudaSetDevice(cutGetAvgGflopsDeviceId());
	//NVMatrix::initDeviceProps();

	this->_w            = new NVMatrix(_filter_channel, _filt_pixs * _in_channel);
	//					NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	this->_bias         = new NVMatrix(_filter_channel, 1);
	//					NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	this->_y               = new NVMatrix(_minibatch_size, \
			_filter_channel * _conv_pixs);
	this->_dE_dy           = new NVMatrix(_y);
	//dE_dx_h是对sigmoid函数的输入求导
	this->_dE_dx_h           = new NVMatrix(_y);
	this->_dE_dw          = new NVMatrix(_w);
	this->_dE_db           = new NVMatrix(_bias);

	this->_w_inc		 = new NVMatrix(_filter_channel, _filt_pixs * _in_channel);
	this->_bias_inc		 = new NVMatrix(_filter_channel, 1);

	//中间变量
	unrolled_x1 = new NVMatrix(_minibatch_size * _conv_pixs, \
			_filt_pixs * _in_channel);
	unranged_y = new NVMatrix(_minibatch_size * _conv_pixs, _filter_channel);
	w_T = new NVMatrix(_w->getNumCols(), _w->getNumRows());
	bias_T = new NVMatrix(_bias->getNumCols(), _bias->getNumRows());	

	unrolled_x2 = new NVMatrix(_filt_pixs * _in_channel, \
			_minibatch_size * _conv_pixs);
	ranged_dE_dx = new NVMatrix(_minibatch_size * _conv_pixs, _filter_channel);
	dE_dw_T = new NVMatrix(_w->getNumCols(), \
			_w->getNumRows());
	dE_db_tmp = new NVMatrix(_minibatch_size, _filter_channel);

	unrolled_conv = new NVMatrix(_minibatch_size * _in_size * _in_size, \
			_filter_size * _filter_size * _filter_channel);
	ranged_w = new NVMatrix(_filter_channel * _filt_pixs, _in_channel);
	unranged_in = new NVMatrix(_minibatch_size * _in_size * _in_size, _in_channel);


	this->_w_inc->zeros();
	this->_bias_inc->zeros();


}

void ConvNet::computeConvOutputs(NVMatrix* _x){

	//100*3*28*28 * 5*5, then add to 100*28*28 * 5*5


	int num_kernel = _minibatch_size * _conv_pixs * _filt_pixs *_in_channel;
	int num_block = num_kernel / 1024 + 1;
	//_x->reValue(32);
	//_w->reValue(1.0f);
	//_bias->reValue(2.0f);
	im2col_filt<<<num_block, 1024>>>(_x->getDevData(), \
			unrolled_x1->getDevData(), num_kernel, _filt_pixs, \
			_filt_pixs * _in_channel, _conv_pixs, _in_size, _in_channel, \
			_filter_size, _out_size, _stride);


	_w->getTranspose(w_T);

	unrolled_x1->rightMult(w_T, 1, unranged_y, handle);

	_bias->getTranspose(bias_T);
	unranged_y->addRowVector(bias_T);

	num_kernel = _minibatch_size * _conv_pixs * _filter_channel;
	num_block = num_kernel / 1024 + 1;
	reshape_y<<<num_block, 1024>>>(unranged_y->getDevData(), _y->getDevData(), \
			num_kernel, _out_size, _filter_channel);
	//unrolled_x1->showValue("data");
	//_w->showValue("whk");
	//_y->showValue("yh");
}


void ConvNet::computeDerivs(NVMatrix* _x){
	//assert(_minibatch_size % 16 == 0);

	//dE_dx_h, 16*16*24*24
	_y->subtractFromScalar(1, _dE_dx_h);

	_dE_dx_h->eltWiseMult(_y);

	_dE_dx_h->eltWiseMult(_dE_dy);

	int num_kernel = _minibatch_size * _conv_pixs * _filt_pixs * _in_channel;
	int num_block = num_kernel / 1024 + 1;

	//_x->reValue(3072);
	//另外一种排列方式，因为需要排列的是24*24的块

	im2col_conv<<<num_block, 1024>>>(_x->getDevData(), \
			unrolled_x2->getDevData(), num_kernel, \
			_conv_pixs, _conv_pixs * _minibatch_size, \
			_filt_pixs, _in_size, _in_channel, _filter_size, \
			_out_size, _stride);	
	//	}
	//_x->showValue("data1");
	//_dE_dx_h->reValue(12544);
	//unrolled_x2->showValue("data");

	num_kernel = _minibatch_size * _conv_pixs * _filter_channel;
	num_block = num_kernel / 1024 + 1;
	reshape_dE_dx_h<<<num_block, 1024>>>(ranged_dE_dx->getDevData(), \
			_dE_dx_h->getDevData(), num_kernel, _out_size, _filter_channel);

	unrolled_x2->rightMult(ranged_dE_dx, 1, dE_dw_T, handle);
	dE_dw_T->getTranspose(_dE_dw);
//ranged_dE_dx->showValue("dedxdh");	
//_dE_dw->showValue("dedwhk");

	blocks = dim3(_minibatch_size, _filter_channel);
	threads = dim3(_out_size, _out_size);
	compute_dE_db<<<blocks, threads, sizeof(float)>>>(_dE_dx_h->getDevData(), \
			dE_db_tmp->getDevData(), _out_size);
	cudaThreadSynchronize();
	dE_db_tmp->sumRow(_dE_db);
}

void ConvNet::computeDerivsOfInput(NVMatrix* dE_dx){

	int num_kernel = _minibatch_size * _in_size * _in_size * _filt_pixs * _filter_channel;
	int num_block = 4096;
//	int num_block = num_kernel / 1024 + 1;

	cudaMemset(unrolled_conv->getDevData(), 0, sizeof(float) * num_kernel);
	im2col_img<<<num_block, 1024>>>(_dE_dx_h->getDevData(), unrolled_conv->getDevData(), \
			num_kernel, _filt_pixs, _filt_pixs * _filter_channel, _in_size, _filter_channel, \
			_in_channel, _filter_size, _out_size, _stride);
	cudaThreadSynchronize();
	num_kernel = _filter_channel * _filt_pixs * _in_channel;

	num_block = num_kernel / 1024 + 1;
	reshape_w<<<num_block, 1024>>>(ranged_w->getDevData(), \
			_w->getDevData(), num_kernel, _filter_size, \
			_filter_channel, _in_channel);
	cudaThreadSynchronize();

	unrolled_conv->rightMult(ranged_w, 1, unranged_in, handle);
	num_kernel = _minibatch_size * _in_size * _in_size * _in_channel;
	num_block = num_kernel / 1024 + 1;
	reshape_In<<<num_block, 1024>>>(dE_dx->getDevData(), unranged_in->getDevData(), \
			num_kernel, _in_size, _in_channel);
	cudaThreadSynchronize();

//t = clock() - t;
//cout << "3: " << ((float)t/CLOCKS_PER_SEC) << " seconds.\n";
//t = clock();
	//_w->showValue("whk");
	//ranged_w->showValue("rangWhk");
//	unranged_in->showValue("unrangIN");
//	unrolled_conv->showValue("");
//	_dE_dx->showValue("dx");

}
/*
void ConvNet::updatePars(){

	_w_inc->addSum(_w, _dE_dw, _momentum, -_weight_decay, \
			-_w_lr / _minibatch_size);
	_w->add(_w_inc, 1, 1);

	_bias_inc->add(_dE_db, _momentum, -_b_lr / _minibatch_size);
	_bias->add(_bias_inc, 1, 1);

}
*/






























