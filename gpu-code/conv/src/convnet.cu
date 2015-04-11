/*
 * filename: convnet.cu
 */
//#include <cutil_inline.h>
#include <time.h>

#include "convnet.cuh"
#include "convnet_kernel.cuh"

using namespace std;
	

ConvNet::ConvNet(pars* netWork){

	this->_numFilters            = netWork->numFilters;

	this->_epsHidVis             = netWork->epsHidVis;
	//hidden bias的learning rate
	this->_epsHidBias            = netWork->epsHidBias;
	//上一次更新的参数控制增长趋势
	this->_mom                   = netWork->mom;
	//hidden原值的参数
	this->_wcHidVis              = netWork->wcHidVis;
	//out原值的参数
	this->_numTrain				 = netWork->trainNum;
	this->_numValid				 = netWork->validNum;
	this->_minibatchSize         = netWork->minibatchSize;
	this->_inSize				 = netWork->inSize;
	this->_filterSize			 = netWork->filterSize;
	this->_stepSize              = netWork->stepSize;
	this->_poolSize              = netWork->poolSize;
	this->_convResultSize		 = (_inSize - _filterSize) / _stepSize + 1;
	this->_poolResultSize		 = this->_convResultSize / _poolSize;
	this->_inChannel			 = netWork->inChannel;
	this->_finePars				 = netWork->finePars;
	this->_filtPixs				 = _filterSize * _filterSize;
	this->_convPixs				 = _convResultSize * _convResultSize;
	cublasCreate(&handle);
}
ConvNet::~ConvNet() {

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
		delete _trainData;
		delete _trainLabel;
		delete _validData;
		delete _validLabel;
		cudaFree(_maxPoolPos);
	cublasDestroy(handle);
}

void ConvNet::initCuda() {
	//cudaSetDevice(cutGetAvgGflopsDeviceId());
	//NVMatrix::initDeviceProps();

	//hidVis大小是16*5*5,bias是5*5
	int inLen = _inChannel * _inSize * _inSize;
	this->_trainData		 = new NVMatrix(_numTrain, inLen);	
	this->_trainLabel		 = new NVMatrix(_numTrain, 1);	
	this->_validData		 = new NVMatrix(_numValid, inLen);	
	this->_validLabel		 = new NVMatrix(_numValid, 1);	
	
	this->_hidVis            = new NVMatrix(_numFilters, _filtPixs);
//					NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	this->_hidBiases         = new NVMatrix(_numFilters, 1);
//					NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	this->_y_h               = new NVMatrix(_minibatchSize, \
			_numFilters * _convPixs);
	this->_y_i               = new NVMatrix(_minibatchSize, \
			_numFilters * _poolResultSize * _poolResultSize);
	this->_dE_dy_i           = new NVMatrix(_y_i);

	this->_dE_dy_h           = new NVMatrix(_y_h);
	this->_dE_dx_h           = new NVMatrix(_y_h);
	this->_dE_dw_hk          = new NVMatrix(_hidVis);
	this->_dE_db_h           = new NVMatrix(_hidBiases);

	this->_hidVisInc		 = new NVMatrix(_numFilters, _filtPixs);
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
	
	//100*3*28*28 * 5*5, then add to 100*28*28 * 5*5

	NVMatrix* unrolledMiniDataMultiChannel = new NVMatrix(_minibatchSize * _convPixs, \
 					_filtPixs * _inChannel);  
	NVMatrix* unrolledMiniData = new NVMatrix(_minibatchSize * _convPixs, \
					_filtPixs);
	NVMatrix* unrangedYH = new NVMatrix(_minibatchSize * _convPixs, _numFilters);
	
	int numKernels = _minibatchSize * _convResultSize * \
			_convResultSize * _filterSize * _filterSize *_inChannel;
	int numBlocks = numKernels / 1024 + 1;
//miniData->reValue(28);
//_hidVis->reValue(9);

	if(_inChannel > 1){
		im2col_filt<<<numBlocks, 1024>>>(miniData->getDevData(), \
				unrolledMiniDataMultiChannel->getDevData(), numKernels, _filtPixs, \
				_filtPixs * _inChannel, _convPixs, _inSize, _inChannel, \
				_filterSize, _convResultSize, _stepSize);

		unrolledMiniData->compactCol(unrolledMiniDataMultiChannel, _inChannel);
	}else{
		im2col_filt<<<numBlocks, 1024>>>(miniData->getDevData(), \
				unrolledMiniData->getDevData(), numKernels, _filtPixs, \
				_filtPixs * _inChannel, _convPixs, _inSize, _inChannel, \
				_filterSize, _convResultSize, _stepSize);
	}
		

	NVMatrix* hidVis_T = new NVMatrix(_hidVis->getNumCols(), _hidVis->getNumRows());
	_hidVis->getTranspose(hidVis_T);

	unrolledMiniData->rightMult(hidVis_T, 1, unrangedYH, handle);

	//每个点都要加上偏置，把偏置当成行向量
	NVMatrix* hidBias_T = new NVMatrix(_hidBiases->getNumCols(), _hidBiases->getNumRows());	
	_hidBiases->getTranspose(hidBias_T);
	unrangedYH->addRowVector(hidBias_T);

	numKernels = _minibatchSize * _convResultSize * _convResultSize * _numFilters;
	numBlocks = numKernels / 1024 + 1;
	reshape_y_h<<<numBlocks, 1024>>>(unrangedYH->getDevData(), _y_h->getDevData(), \
					numKernels, _convResultSize, _numFilters);
//unrolledMiniData->showValue("data");
//_hidVis->showValue("whk");
//_y_h->showValue("yh");
	

	delete unrolledMiniDataMultiChannel;
	delete unrolledMiniData;
	delete unrangedYH;
	delete hidVis_T;
	delete hidBias_T;
}

/*
void ConvNet::computeAvgOutputs(){
	//16*16
	dim3 blocks = dim3(_minibatchSize, _numFilters);
	dim3 threads = dim3(_poolResultSize, _poolResultSize);
	//24*24,pooling到12*12
	avg_pooling<<<blocks, threads>>>(_y_h->getDevData(), _y_i->getDevData());
	cudaThreadSynchronize();
}*/

void ConvNet::computeMaxOutputs(){
	//16*16
	dim3 blocks = dim3(_minibatchSize, _numFilters);
	dim3 threads = dim3(ceil(_poolResultSize / 16.0) * 16,  ceil(_poolResultSize / 16.0) * 16);
	//24*24,pooling到12*12
//_y_h->reValue(20);
//_y_h->showValue("yh");
	max_pooling<<<blocks, threads, sizeof(float) * _convResultSize * _convResultSize>>>(_y_h->getDevData(), \
			_y_i->getDevData(), _maxPoolPos, _convResultSize, _poolResultSize, _poolSize);	
//_y_i->showValue("yi");
	cudaThreadSynchronize();
}


void ConvNet::computeDerivs(NVMatrix* miniData){
	//assert(_minibatchSize % 16 == 0);

	//dE_dy_i, 16*16*12*12
	//每次还原一个点，因为四个点只需还原一个，因此只用12*12的线程做
	dim3 blocks = dim3(_minibatchSize, _numFilters);
	dim3 threads = dim3(ceil(_poolResultSize / 16.0) * 16,  ceil(_poolResultSize / 16.0) * 16);
	//dE_dy_h, 16*16*24*24
	_dE_dy_h->zeros();
	compute_dE_dy_h_max<<<blocks, threads>>>(_dE_dy_i->getDevData(), \
			_dE_dy_h->getDevData(), _maxPoolPos, _convResultSize, \
			_poolResultSize, _poolSize);
	cudaThreadSynchronize();

	//dE_dx_h, 16*16*24*24
	_y_h->subtractFromScalar(1, _dE_dx_h);

	_dE_dx_h->eltWiseMult(_y_h);

	_dE_dx_h->eltWiseMult(_dE_dy_h);

	int numKernels = _minibatchSize * _convPixs * _filtPixs * _inChannel;
	int numBlocks = numKernels / 1024 + 1;

//miniData->reValue(28);
	//另外一种排列方式，因为需要排列的是24*24的块
	NVMatrix* unrolledMiniDataMultiChannel = new NVMatrix(_filtPixs, \
				_minibatchSize * _convPixs * _inChannel);
	NVMatrix* unrolledMiniData = new NVMatrix(_filtPixs, \
				_minibatchSize * _convPixs);
	NVMatrix* rangedDEDXH = new NVMatrix(_minibatchSize * _convPixs, _numFilters);
	NVMatrix* dE_dw_hk_T = new NVMatrix(_hidVis->getNumCols(), \
				_hidVis->getNumRows());

	if(_inChannel > 1){
		im2col_conv<<<numBlocks, 1024>>>(miniData->getDevData(), \
				unrolledMiniDataMultiChannel->getDevData(), numKernels, \
				_convPixs * _inChannel, _convPixs, \
				_convPixs * _minibatchSize * _inChannel, _filtPixs, \
				_inSize, _inChannel, _filterSize, _convResultSize, _stepSize);	
		unrolledMiniData->compactCol(unrolledMiniDataMultiChannel, _inChannel);
	}else{
		im2col_conv<<<numBlocks, 1024>>>(miniData->getDevData(), \
				unrolledMiniData->getDevData(), numKernels, \
				_convPixs * _inChannel, _convPixs * _minibatchSize, \
				_convPixs * _minibatchSize * _inChannel, _filtPixs, \
				_inSize, _inChannel, _filterSize, _convResultSize, _stepSize);	
	}
//miniData->showValue("data1");
//_dE_dx_h->reValue(20);
//unrolledMiniDataMultiChannel->showValue("data");
//unrolledMiniData->showValue("data");

	numKernels = _minibatchSize * _convPixs * _numFilters;
	numBlocks = numKernels / 1024 + 1;
	reshape_dE_dx_h<<<numBlocks, 1024>>>(rangedDEDXH->getDevData(), \
			_dE_dx_h->getDevData(), numKernels, _convResultSize, _numFilters);
	
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
			dE_db_h_tmp->getDevData(), _convResultSize);
	cudaThreadSynchronize();
	dE_db_h_tmp->sumRow(_dE_db_h);

	delete dE_db_h_tmp;

}

void ConvNet::updatePars(){

	_hidVisInc->addSum(_hidVis, _dE_dw_hk, _mom, -_wcHidVis, \
			-_epsHidVis / _minibatchSize);
	_hidVis->add(_hidVisInc, 1, 1);

	_hidBiasInc->add(_dE_db_h, _mom, -_epsHidBias / _minibatchSize);
	_hidBiases->add(_hidBiasInc, 1, 1);

}































