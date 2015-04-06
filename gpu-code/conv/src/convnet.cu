/*
 * filename: convnet.cu
 */
//#include <cutil_inline.h>
#include <time.h>

#include "convnet.cuh"
#include "convnet_kernel.cuh"

using namespace std;
	

ConvNet::ConvNet(Matrix* hHidVis, Matrix* hHidBiases, pars* netWork){

	this->_numFilters            = hHidVis->getNumRows();

	this->_hHidVis               = hHidVis;
	this->_hHidBiases            = hHidBiases;

	this->_epsHidVis             = netWork->epsHidVis;
	//hidden bias的learning rate
	this->_epsHidBias            = netWork->epsHidBias;
	//上一次更新的参数控制增长趋势
	this->_mom                   = netWork->mom;
	//hidden原值的参数
	this->_wcHidVis              = netWork->wcHidVis;
	//out原值的参数
	this->_minibatchSize         = netWork->minibatchSize;
	this->_inSize				 = netWork->inSize;
	this->_filterSize			 = netWork->filterSize;
	this->_stepSize              = netWork->stepSize;
	this->_convResultSize		 = _inSize - _filterSize + 1;
	this->_poolResultSize		 = this->_convResultSize / AVG_POOL_X;
	this->_inChannel			 = netWork->inChannel;
	cublasCreate(&handle);
}
ConvNet::~ConvNet() {
		delete _hHidVis;
		delete _hHidBiases;

		delete _hidVis;
		delete _hidVisInc;
		delete _hidBiases;
		delete _hidBiasInc;

		delete _y_h; 
		delete  _y_i; 
		delete _dE_dy_i;
		delete _dE_dy_h;
		delete _dE_dx_h;
		delete _dE_dw_hk;
		delete _dE_db_h;
		cudaFree(_maxPoolPos);
	cublasDestroy(handle);
}

void ConvNet::initCuda() {
	//cudaSetDevice(cutGetAvgGflopsDeviceId());
	//NVMatrix::initDeviceProps();

	//hidVis大小是16*5*5,bias是5*5
	
	int filtPixs = _filterSize * _filterSize;
	int convPixs = _convResultSize * _convResultSize;
	
	this->_hidVis            = new NVMatrix(_hHidVis, true);
//					NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	this->_hidBiases         = new NVMatrix(_hHidBiases, true);
//					NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	this->_y_h               = new NVMatrix(_minibatchSize, \
			_numFilters * convPixs);
	this->_y_i               = new NVMatrix(_minibatchSize, \
			_numFilters * _poolResultSize * _poolResultSize);
	this->_dE_dy_i           = new NVMatrix(_y_i);

	this->_dE_dy_h           = new NVMatrix(_y_h);
	this->_dE_dx_h           = new NVMatrix(_y_h);
	this->_dE_dw_hk          = new NVMatrix(_hidVis);
	this->_dE_db_h           = new NVMatrix(_hidBiases);

	this->_hidVisInc		 = new NVMatrix(_numFilters, filtPixs);
	this->_hidBiasInc		 = new NVMatrix(_numFilters, 1);
	

	this->_hidVisInc->zeros();
	this->_hidBiasInc->zeros();
	
	cudaError_t status = cudaMalloc((void**) &_maxPoolPos, \
			_minibatchSize * _numFilters * _poolResultSize * _poolResultSize * sizeof(int));
    if (status != cudaSuccess) {
		fprintf(stderr, "!!!! device memory allocation error\n");
        exit(EXIT_FAILURE);
	}

}

void ConvNet::computeConvOutputs(NVMatrix* miniData){
	
	int filtPixs = _filterSize * _filterSize;
	int convPixs = _convResultSize * _convResultSize;
	//100*3*28*28 * 5*5, then add to 100*28*28 * 5*5
	NVMatrix* unrolledMiniDataMultiChannel = new NVMatrix(_minibatchSize *convPixs, \
 					filtPixs * _inChannel);  
	NVMatrix* unrolledMiniData = new NVMatrix(_minibatchSize * convPixs, \
					filtPixs);
	NVMatrix* unrangedYH = new NVMatrix(_minibatchSize * convPixs, _numFilters);
	
	int numKernels = _minibatchSize * _convResultSize * \
			_convResultSize * _filterSize * _filterSize *_inChannel;
	int numBlocks = numKernels / 1024 + 1;
	im2col_filt<<<numBlocks, 1024>>>(miniData->getDevData(), \
			unrolledMiniDataMultiChannel->getDevData(), numKernels, filtPixs, \
			filtPixs * _inChannel, convPixs);

	unrolledMiniData->compactCol(unrolledMiniDataMultiChannel, _inChannel);

	NVMatrix* hidVis_T = new NVMatrix(_hidVis->getNumCols(), _hidVis->getNumRows());
	_hidVis->getTranspose(hidVis_T);

	unrolledMiniData->rightMult(hidVis_T, 1, unrangedYH, handle);
	numKernels = _minibatchSize * _convResultSize * _convResultSize * _numFilters;
	numBlocks = numKernels / 1024 + 1;
	reshape_y_h<<<numBlocks, 1024>>>(unrangedYH->getDevData(), _y_h->getDevData(), \
					numKernels);
	delete unrolledMiniDataMultiChannel;
	delete unrolledMiniData;
	delete unrangedYH;
	delete hidVis_T;
}
void ConvNet::computeAvgOutputs(){
	//16*16
	dim3 blocks = dim3(_minibatchSize, _numFilters);
	dim3 threads = dim3(_poolResultSize, _poolResultSize);
	//24*24,pooling到12*12
	avg_pooling<<<blocks, threads>>>(_y_h->getDevData(), _y_i->getDevData());
	cudaThreadSynchronize();
}

void ConvNet::computeMaxOutputs(){
	//16*16
	dim3 blocks = dim3(_minibatchSize, _numFilters);
	dim3 threads = dim3(_poolResultSize, _poolResultSize);
	//24*24,pooling到12*12
	max_pooling<<<blocks, threads>>>(_y_h->getDevData(), _y_i->getDevData(), \
			_maxPoolPos);	

	cudaThreadSynchronize();
}

void ConvNet::computeDerivs(NVMatrix* miniData, NVMatrix* dE_dy_j, NVMatrix* avgOut){
	//assert(_minibatchSize % 16 == 0);

	//dE_dy_i, 16*16*12*12
	NVMatrix* avgOut_T = new NVMatrix(avgOut->getNumCols(), avgOut->getNumRows());
	avgOut->getTranspose(avgOut_T);
	dE_dy_j->rightMult(avgOut_T, 1, _dE_dy_i, handle);
	//每次还原一个点，因为四个点只需还原一个，因此只用12*12的线程做
	dim3 blocks = dim3(_minibatchSize, _numFilters);
	dim3 threads = dim3(ceil(_poolResultSize / 16.0) * 16,  ceil(_poolResultSize / 16.0) * 16);
	//dE_dy_h, 16*16*24*24
	_dE_dy_h->zeros();
	compute_dE_dy_h_max<<<blocks, threads>>>(_dE_dy_i->getDevData(), \
			_dE_dy_h->getDevData(), _maxPoolPos);
	cudaThreadSynchronize();

	//dE_dx_h, 16*16*24*24
	_y_h->subtractFromScalar(1, _dE_dx_h);

	_dE_dx_h->eltWiseMult(_y_h);

	_dE_dx_h->eltWiseMult(_dE_dy_h);

	
	int filtPixs = _filterSize * _filterSize;
	int convPixs = _convResultSize * _convResultSize;

	int numKernels = _minibatchSize * convPixs * filtPixs * _inChannel;
	int numBlocks = numKernels / 1024 + 1;

	//另外一种排列方式，因为需要排列的是24*24的块
	NVMatrix* unrolledMiniDataMultiChannel = new NVMatrix(filtPixs, \
				_minibatchSize * convPixs * _inChannel);
	NVMatrix* unrolledMiniData = new NVMatrix(filtPixs, \
				_minibatchSize * convPixs);
	NVMatrix* rangedDEDXH = new NVMatrix(_minibatchSize * convPixs, _numFilters);
	NVMatrix* dE_dw_hk_T = new NVMatrix(_hidVis->getNumCols(), \
				_hidVis->getNumRows());

	im2col_conv<<<numBlocks, 1024>>>(miniData->getDevData(), \
			unrolledMiniDataMultiChannel->getDevData(), numKernels, \
			convPixs * _inChannel, convPixs * _minibatchSize, \
			convPixs * _minibatchSize * _inChannel, filtPixs);	
	unrolledMiniData->compactCol(unrolledMiniDataMultiChannel, _inChannel);


	numKernels = _minibatchSize * convPixs * _numFilters;
	numBlocks = numKernels / 1024 + 1;
	reshape_dE_dx_h<<<numBlocks, 1024>>>(rangedDEDXH->getDevData(), \
			_dE_dx_h->getDevData(), numKernels);
	
	unrolledMiniData->rightMult(rangedDEDXH, 1, dE_dw_hk_T, handle);
	dE_dw_hk_T->getTranspose(_dE_dw_hk);
//rangedDEDXH->showValue("dedxdh");	
//_dE_dw_hk->showValue("dedwhk");
	delete rangedDEDXH;
	delete dE_dw_hk_T;
	delete unrolledMiniDataMultiChannel;
	delete unrolledMiniData;
	
	NVMatrix* dE_db_h_tmp = new NVMatrix(_minibatchSize, _numFilters);
	blocks = dim3(_minibatchSize, _numFilters);
	threads = dim3(_convResultSize, _convResultSize);
	compute_dE_db_h<<<blocks, threads, sizeof(float)>>>(_dE_dx_h->getDevData(), \
			dE_db_h_tmp->getDevData());
	cudaThreadSynchronize();
	dE_db_h_tmp->sumRow(_dE_db_h);

	delete avgOut_T;
	delete dE_db_h_tmp;

}

void ConvNet::updatePars(){

	_hidVisInc->addSum(_hidVis, _dE_dw_hk, _mom, -_wcHidVis, \
			-_epsHidVis / _minibatchSize);
	_hidVis->add(_hidVisInc, 1, 1);

	_hidBiasInc->add(_dE_db_h, _mom, -_epsHidBias / _minibatchSize);
	_hidBiases->add(_hidBiasInc, 1, 1);

}































