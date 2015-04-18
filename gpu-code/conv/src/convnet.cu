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

	delete unrolledMiniData1;
	delete unrangedYH;
	delete hidVis_T;
	delete hidBias_T;
	delete unrolledMiniData2;
	delete rangedDEDXH;
	delete dE_dw_hk_T;
	delete dE_db_h_tmp;
	delete unrolledConv;
	delete rangedHidVis;
	delete unrangedIn;

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

	this->_hidVis            = new NVMatrix(_numFilters, _filtPixs * _inChannel);
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

	this->_hidVisInc		 = new NVMatrix(_numFilters, _filtPixs * _inChannel);
	this->_hidBiasInc		 = new NVMatrix(_numFilters, 1);

	//中间变量
	unrolledMiniData1 = new NVMatrix(_minibatchSize * _convPixs, \
			_filtPixs * _inChannel);
	unrangedYH = new NVMatrix(_minibatchSize * _convPixs, _numFilters);
	hidVis_T = new NVMatrix(_hidVis->getNumCols(), _hidVis->getNumRows());
	hidBias_T = new NVMatrix(_hidBiases->getNumCols(), _hidBiases->getNumRows());	

	unrolledMiniData2 = new NVMatrix(_filtPixs * _inChannel, \
			_minibatchSize * _convPixs);
	rangedDEDXH = new NVMatrix(_minibatchSize * _convPixs, _numFilters);
	dE_dw_hk_T = new NVMatrix(_hidVis->getNumCols(), \
			_hidVis->getNumRows());
	dE_db_h_tmp = new NVMatrix(_minibatchSize, _numFilters);

	unrolledConv = new NVMatrix(_minibatchSize * _inSize * _inSize, \
			_filterSize * _filterSize * _numFilters);
	rangedHidVis = new NVMatrix(_numFilters * _filtPixs, _inChannel);
	unrangedIn = new NVMatrix(_minibatchSize * _inSize * _inSize, _inChannel);


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


	int numKernels = _minibatchSize * _convPixs * _filtPixs *_inChannel;
	int numBlocks = numKernels / 1024 + 1;
	//miniData->reValue(32);
	//_hidVis->reValue(1.0f);
	//_hidBiases->reValue(2.0f);
	im2col_filt<<<numBlocks, 1024>>>(miniData->getDevData(), \
			unrolledMiniData1->getDevData(), numKernels, _filtPixs, \
			_filtPixs * _inChannel, _convPixs, _inSize, _inChannel, \
			_filterSize, _convResultSize, _stepSize);


	_hidVis->getTranspose(hidVis_T);

	unrolledMiniData1->rightMult(hidVis_T, 1, unrangedYH, handle);

	_hidBiases->getTranspose(hidBias_T);
	unrangedYH->addRowVector(hidBias_T);

	numKernels = _minibatchSize * _convPixs * _numFilters;
	numBlocks = numKernels / 1024 + 1;
	reshape_y_h<<<numBlocks, 1024>>>(unrangedYH->getDevData(), _y_h->getDevData(), \
			numKernels, _convResultSize, _numFilters);
	//unrolledMiniData1->showValue("data");
	//_hidVis->showValue("whk");
	//_y_h->showValue("yh");
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
	max_pooling<<<blocks, threads, sizeof(float) * _convResultSize * _convResultSize>>>(_y_h->getDevData(), \
			_y_i->getDevData(), _maxPoolPos, _convResultSize, _poolResultSize, _poolSize);	
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

	//miniData->reValue(3072);
	//另外一种排列方式，因为需要排列的是24*24的块

	im2col_conv<<<numBlocks, 1024>>>(miniData->getDevData(), \
			unrolledMiniData2->getDevData(), numKernels, \
			_convPixs, _convPixs * _minibatchSize, \
			_filtPixs, _inSize, _inChannel, _filterSize, \
			_convResultSize, _stepSize);	
	//	}
	//miniData->showValue("data1");
	//_dE_dx_h->reValue(12544);
	//unrolledMiniData2->showValue("data");

	numKernels = _minibatchSize * _convPixs * _numFilters;
	numBlocks = numKernels / 1024 + 1;
	reshape_dE_dx_h<<<numBlocks, 1024>>>(rangedDEDXH->getDevData(), \
			_dE_dx_h->getDevData(), numKernels, _convResultSize, _numFilters);

	unrolledMiniData2->rightMult(rangedDEDXH, 1, dE_dw_hk_T, handle);
	dE_dw_hk_T->getTranspose(_dE_dw_hk);
//rangedDEDXH->showValue("dedxdh");	
//_dE_dw_hk->showValue("dedwhk");

	blocks = dim3(_minibatchSize, _numFilters);
	threads = dim3(_convResultSize, _convResultSize);
	compute_dE_db_h<<<blocks, threads, sizeof(float)>>>(_dE_dx_h->getDevData(), \
			dE_db_h_tmp->getDevData(), _convResultSize);
	cudaThreadSynchronize();
	dE_db_h_tmp->sumRow(_dE_db_h);
}

void ConvNet::computeDerivsToIn(NVMatrix* downLayer_dE_dy_i){

	int numKernels = _minibatchSize * _inSize * _inSize * _filtPixs * _numFilters;
	int numBlocks = 4096;
//	int numBlocks = numKernels / 1024 + 1;

	cudaMemset(unrolledConv->getDevData(), 0, sizeof(float) * numKernels);
//clock_t t = clock();
	im2col_img<<<numBlocks, 1024>>>(_dE_dx_h->getDevData(), unrolledConv->getDevData(), \
			numKernels, _filtPixs, _filtPixs * _numFilters, _inSize, _numFilters, \
			_inChannel, _filterSize, _convResultSize, _stepSize);
	cudaThreadSynchronize();
	//	_dE_dx_h->showValue("dedxh");
	
//t = clock() - t;
//cout << "1: " << ((float)t/CLOCKS_PER_SEC) << " seconds.\n";
//t = clock();



	//unrolledConv->showValue("");
	numKernels = _numFilters * _filtPixs * _inChannel;
	numBlocks = numKernels / 1024 + 1;
	reshape_hidVis<<<numBlocks, 1024>>>(rangedHidVis->getDevData(), \
			_hidVis->getDevData(), numKernels, _filterSize, \
			_numFilters, _inChannel);
	cudaThreadSynchronize();

	unrolledConv->rightMult(rangedHidVis, 1, unrangedIn, handle);
	numKernels = _minibatchSize * _inSize * _inSize * _inChannel;
	numBlocks = numKernels / 1024 + 1;
	reshape_In<<<numBlocks, 512>>>(downLayer_dE_dy_i->getDevData(), unrangedIn->getDevData(), \
			numKernels, _inSize, _inChannel);
	cudaThreadSynchronize();

//t = clock() - t;
//cout << "3: " << ((float)t/CLOCKS_PER_SEC) << " seconds.\n";
//t = clock();
	//_hidVis->showValue("whk");
	//rangedHidVis->showValue("rangWhk");
//	unrangedIn->showValue("unrangIN");
//	unrolledConv->showValue("");
//	_dE_dx->showValue("dx");

}

void ConvNet::updatePars(){

	_hidVisInc->addSum(_hidVis, _dE_dw_hk, _mom, -_wcHidVis, \
			-_epsHidVis / _minibatchSize);
	_hidVis->add(_hidVisInc, 1, 1);

	_hidBiasInc->add(_dE_db_h, _mom, -_epsHidBias / _minibatchSize);
	_hidBiases->add(_hidBiasInc, 1, 1);

}































