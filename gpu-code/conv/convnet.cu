/*
 * filename: convnet.cu
 */
//#include <cutil_inline.h>
#include <time.h>

#include "convnet.cuh"
#include "convnet_kernel.cuh"

using namespace std;
	

ConvNet::ConvNet(Matrix* hHidVis, Matrix* hAvgOut, Matrix* hHidBiases, \
		Matrix*	hOutBiases,	float epsHidVis, float epsAvgOut, float epsHidBias, \
		float epsOutBias, float mom, float wcHidVis, float wcAvgOut, \
		const int minibatchSize, const int inSize, \
		const int filterSize, const int inChannel, \
		const int numFilters){

	this->_numFilters            = hHidVis->getNumRows();
	this->_numOut                = hAvgOut->getNumCols();

	this->_hHidVis               = hHidVis;
	this->_hAvgOut               = hAvgOut;
	this->_hHidBiases            = hHidBiases;
	this->_hOutBiases            = hOutBiases;

	//w_ij的learning rate
	this->_epsHidVis             = epsHidVis;
	//w_hk的learning rate
	this->_epsAvgOut             = epsAvgOut;
	//hidden bias的learning rate
	this->_epsHidBias            = epsHidBias;
	//out bias learning rate
	this->_epsOutBias            = epsOutBias;
	//上一次更新的参数控制增长趋势
	this->_mom                   = mom;
	//hidden原值的参数
	this->_wcHidVis              = wcHidVis;
	//out原值的参数
	this->_wcAvgOut              = wcAvgOut;
	this->_minibatchSize         = minibatchSize;
	this->_inSize				 = inSize;
	this->_filterSize			 = filterSize;
	this->_convResultSize		 = inSize - filterSize + 1;
	this->_poolResultSize		 = this->_convResultSize / AVG_POOL_X;
	this->_inChannel			 = inChannel;
}
ConvNet::~ConvNet() {
	/*	delete _hHidVis;
		delete _hHidVisInc;
		delete _hHidBiases;
		delete _hHidBiasInc;
		delete _hAvgOut;
		delete _hAvgOutInc;
		delete _hOutBiases;
		delete _hOutBiasInc;

		delete _hidVis;
		delete _hidVisInc;
		delete _hidBiases;
		delete _hidBiasInc;
		delete _avgOut;
		delete _avgOutInc;
		delete _outBiases;
		delete _outBiasInc;

		delete _y_h; 
		delete  _y_i; 
		delete  _y_j; 
		delete  _dE_dy_j;
		delete _dE_db_j;
		delete _dE_dw_ij;
		delete _dE_dy_i;
		delete _dE_dy_h;
		delete _dE_dx_h;
		delete _dE_dw_hk;
		delete _dE_db_h;
	 */
	cublasDestroy(handle);
}

void ConvNet::initCuda() {
	//cudaSetDevice(cutGetAvgGflopsDeviceId());
	//NVMatrix::initDeviceProps();

	//hidVis大小是16*5*5,bias是5*5
	this->_hidVis            = new NVMatrix(_hHidVis, true);
	this->_avgOut            = new NVMatrix(_hAvgOut, true);
	this->_hidBiases         = new NVMatrix(_hHidBiases, true);
	this->_outBiases         = new NVMatrix(_hOutBiases, true);

	this->_y_h               = new NVMatrix(_minibatchSize, \
			_numFilters * _convResultSize * _convResultSize);
	this->_y_i               = new NVMatrix(_minibatchSize, \
			_numFilters * _poolResultSize * _poolResultSize);
	this->_y_j               = new NVMatrix(_minibatchSize, _numOut);

	this->_dE_dy_j           = new NVMatrix(_y_j);
	this->_dE_db_j           = new NVMatrix(_outBiases);
	this->_dE_dw_ij          = new NVMatrix(_avgOut);
	this->_dE_dy_i           = new NVMatrix(_y_i);

	this->_dE_dy_h           = new NVMatrix(_y_h);
	this->_dE_dx_h           = new NVMatrix(_y_h);
	this->_dE_dw_hk          = new NVMatrix(_hidVis);
	this->_dE_db_h           = new NVMatrix(_hidBiases);

	this->_avgOutInc		 = new NVMatrix(_avgOut);

	this->_outBiasInc		 = new NVMatrix(1, _numOut);
	this->_hidVisInc		 = new NVMatrix(_numFilters, _filterSize * _filterSize);
	this->_hidBiasInc		 = new NVMatrix(_numFilters, 1);
	this->_avgOutInc->zeros();
	this->_outBiasInc->zeros();
	this->_hidVisInc->zeros();
	this->_hidBiasInc->zeros();
	
	cudaError_t status = cudaMalloc((void**) &_maxPoolPos, \
			_minibatchSize * _numFilters * _poolResultSize * _poolResultSize * sizeof(int));
    if (status != cudaSuccess) {
		fprintf(stderr, "!!!! device memory allocation error\n");
        exit(EXIT_FAILURE);
	}

	cublasCreate(&handle);

}

void ConvNet::computeConvOutputs(NVMatrix* miniData){
	//16*16
	dim3 blocks = dim3(_minibatchSize, _numFilters);
	//28*5，此处需要改变，低效
	dim3 threads = dim3(_convResultSize, _convResultSize);
	int filConvtimes = _filterSize / _convResultSize;
	int imgConvtimes = _inSize / _convResultSize;
	convolution_forward<<<blocks, threads>>>(miniData->getDevData(), \
			_hidVis->getDevData(), _hidBiases->getDevData(), _y_h->getDevData(), \
			filConvtimes, imgConvtimes);
	cudaThreadSynchronize();
	//	cutilCheckMsg("Kernel execution failed");
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

void ConvNet::computeClassOutputs(){
	int blocks = _minibatchSize;
	int threads = _numOut;
	_y_i->rightMult(_avgOut, 1, _y_j, handle);

	_y_j->addRowVector(_outBiases, blocks, threads);
	//防止_y_j溢出，保证E^x，控制在e^-15~e^15

	_y_j->apply(NVMatrix::SOFTMAX, blocks, 1);
//	NVMatrix* y_j_sum = _y_j->sumCol(blocks, threads);
//	_y_j->eltWiseDivideByVector(y_j_sum, blocks, threads);
//delete y_j_sum;

}

double ConvNet::computeError(const NVMatrix* miniLabels, int& numError){
	int label;
	Matrix* hlabels = new Matrix(miniLabels->getNumRows(), miniLabels->getNumCols());
	miniLabels->copyToHost(hlabels);
	Matrix* y_j_CPU = new Matrix(_y_j->getNumRows(), _y_j->getNumCols());
	_y_j->copyToHost(y_j_CPU);
	Matrix* correctProbs = new Matrix(_y_j->getNumRows(), 1); 
	for (int c = 0; c < _y_j->getNumRows(); c++) {
		label = hlabels->getCell(c, 0);
		correctProbs->getCell(c, 0) = y_j_CPU->getCell(c, label);

		int y_j_max_pos = 0;
		for(int j = 1; j < _y_j->getNumCols(); j++){
			if(y_j_CPU->getCell(c, y_j_max_pos) < y_j_CPU->getCell(c, j))
				y_j_max_pos = j;
		}
	//	cout << y_j_max_pos << ":" << label << " ";
		if(y_j_max_pos != label)
			numError++;
	}   
//	cout << endl;
	correctProbs->apply(Matrix::LOG);
	double result = -correctProbs->sum();
	cudaThreadSynchronize();

	delete hlabels;
	delete y_j_CPU;
	delete correctProbs;
	return result;
}

void ConvNet::computeDerivs(NVMatrix* miniData, NVMatrix* miniLabels){
	assert(_minibatchSize % 16 == 0);
	assert(miniLabels->getNumRows() == miniData->getNumRows());

	compute_dE_dy_j<<<_minibatchSize, _numOut>>>(_y_j->getDevData(), \
			miniLabels->getDevData(), _dE_dy_j->getDevData());
	cudaThreadSynchronize();

	NVMatrix* y_i_T = new NVMatrix(_y_i->getNumCols(), _y_i->getNumRows());
	_y_i->getTranspose(y_i_T);
	//16*12*12 * 16, 16 * 10
	y_i_T->rightMult(_dE_dy_j, 1, _dE_dw_ij, handle);
/*
t = clock() - t;
cout << "dEdwij: " << (float)t/CLOCKS_PER_SEC << " seconds. \n";
t = clock();
*/
	//16 * 10
	_dE_dy_j->sumRow(_dE_db_j, _minibatchSize, _numOut);

/*
t = clock() - t;
cout << "dEdbj: " << (float)t/CLOCKS_PER_SEC << " seconds. \n";
t = clock();
*/

	//dE_dy_i, 16*16*12*12
	NVMatrix* avgOut_T = new NVMatrix(_avgOut->getNumCols(), _avgOut->getNumRows());
	_avgOut->getTranspose(avgOut_T);
	_dE_dy_j->rightMult(avgOut_T, 1, _dE_dy_i, handle);


	//每次还原一个点，因为四个点只需还原一个，因此只用12*12的线程做
	dim3 blocks = dim3(_minibatchSize, _numFilters);
	dim3 threads = dim3(ceil(_poolResultSize / 16.0) * 16,  ceil(_poolResultSize / 16.0) * 16);
	//dE_dy_h, 16*16*24*24
/*
t = clock() - t;
cout << "dEdyi: " << (float)t/CLOCKS_PER_SEC << " seconds. \n";
t = clock();
*/

	_dE_dy_h->zeros();
	compute_dE_dy_h_max<<<blocks, threads>>>(_dE_dy_i->getDevData(), \
			_dE_dy_h->getDevData(), _maxPoolPos);
	cudaThreadSynchronize();
/*
t = clock() - t;
cout << "dEdyi: " << (float)t/CLOCKS_PER_SEC << " seconds. \n";
t = clock();
*/

	//dE_dx_h, 16*16*24*24
	_y_h->subtractFromScalar(1, _dE_dx_h, _minibatchSize * _numFilters, \
			_convResultSize * _convResultSize);

	_dE_dx_h->eltWiseMult(_y_h, _minibatchSize * _numFilters, \
			_convResultSize * _convResultSize);

	_dE_dx_h->eltWiseMult(_dE_dy_h, _minibatchSize * _numFilters, \
			_convResultSize * _convResultSize);

clock_t t = clock();
	NVMatrix* dE_dw_hk_tmp = new NVMatrix(_minibatchSize, \
			_numFilters * _filterSize *_filterSize);
	blocks = dim3(_minibatchSize, _numFilters);
	threads = dim3(_filterSize, _filterSize);
	int filConvtimes = _convResultSize / _filterSize;
	int imgConvtimes = _inSize / _filterSize;
	convolution_backward<<<blocks, threads>>>(miniData->getDevData(), \
			_dE_dx_h->getDevData(), dE_dw_hk_tmp->getDevData(), \
			filConvtimes, imgConvtimes);
	cudaThreadSynchronize();

t = clock() - t;
cout << "dEdwhktmp: " << (float)t/CLOCKS_PER_SEC << " seconds. \n";
t = clock();

	//按每一列作为一个线程，故两者乘积要比16*24*24大
	dE_dw_hk_tmp->sumRow(_dE_dw_hk, _numFilters * 4, _minibatchSize * 16);
	
	NVMatrix* dE_db_h_tmp = new NVMatrix(_minibatchSize, _numFilters);
	blocks = dim3(_minibatchSize, _numFilters);
	threads = dim3(_convResultSize, _convResultSize);
	compute_dE_db_h<<<blocks, threads, sizeof(float)>>>(_dE_dx_h->getDevData(), \
			dE_db_h_tmp->getDevData());
	cudaThreadSynchronize();
/*
t = clock() - t;
cout << "dEdbhtmp: " << (float)t/CLOCKS_PER_SEC << " seconds. \n";
t = clock();
*/
	dE_db_h_tmp->sumRow(_dE_db_h, _numFilters, _minibatchSize);


	delete y_i_T;
	delete avgOut_T;
	delete dE_dw_hk_tmp;
	delete dE_db_h_tmp;

}

void ConvNet::updatePars(){
	
	int numBlocks = _numFilters * _numOut;
	int numThreadsPerBlock = _poolResultSize * _poolResultSize;
	_avgOutInc->addSum(_avgOut, _dE_dw_ij, _mom, -_wcAvgOut, \
			-_epsAvgOut / _minibatchSize, numBlocks, numThreadsPerBlock);

	_avgOut->add(_avgOutInc, 1, 1, numBlocks, numThreadsPerBlock);

	numBlocks = 1;
	numThreadsPerBlock = _numOut;
	_outBiasInc->add(_dE_db_j, _mom, -_epsHidBias / _minibatchSize, numBlocks, \
			numThreadsPerBlock);
	_outBiases->add(_outBiasInc, 1, 1, numBlocks, numThreadsPerBlock);

	numBlocks = _numFilters;
	numThreadsPerBlock = _filterSize * _filterSize;
	_hidVisInc->addSum(_hidVis, _dE_dw_hk, _mom, -_wcHidVis, \
			-_epsHidVis / _minibatchSize, numBlocks, numThreadsPerBlock);
	_hidVis->add(_hidVisInc, 1, 1, numBlocks, numThreadsPerBlock);


	numThreadsPerBlock = 1;
	_hidBiasInc->add(_dE_db_h, _mom, -_epsHidBias / _minibatchSize, numBlocks, \
			numThreadsPerBlock);
	_hidBiases->add(_hidBiasInc, 1, 1, numBlocks, numThreadsPerBlock);

}

void ConvNet::computeLogistic(NVMatrix* miniData, NVMatrix* miniLabels, bool isTrain){

//	clock_t t = clock();
	int block = _minibatchSize;
	int thread = _numOut;

	miniData->rightMult(_avgOut, 1, _y_j, handle);
	
//	t = clock() - t;                                                                                        
//		cout << "rightmulti1: " << (float)t/CLOCKS_PER_SEC << " seconds. \n";
//		t = clock();

	_y_j->addRowVector(_outBiases, block, thread);
//	t = clock() - t;                                                                
//	cout << "yj: " << (float)t/CLOCKS_PER_SEC << " seconds. \n";
//	t = clock();

	_y_j->apply(NVMatrix::SOFTMAX, block, 1);

	compute_dE_dy_j<<<block, thread>>>(_y_j->getDevData(), \
			miniLabels->getDevData(), _dE_dy_j->getDevData());
//	t = clock() - t;                                                                
//	cout << "yj: " << (float)t/CLOCKS_PER_SEC << " seconds. \n";
//	t = clock();

	NVMatrix* y_i_T = new NVMatrix(miniData->getNumCols(), miniData->getNumRows());
	miniData->getTranspose(y_i_T);

	//16*12*12 * 16, 16 * 10
	y_i_T->rightMult(_dE_dy_j, 1, _dE_dw_ij, handle);

	//16 * 10
	_dE_dy_j->sumRow(_dE_db_j, _minibatchSize, _numOut);
	if(isTrain){
		int numBlocks = _numFilters;
		int numThreadsPerBlock = _numOut;
		_avgOutInc->addSum(_avgOut, _dE_dw_ij, _mom, -_wcAvgOut, \
				-_epsAvgOut / _minibatchSize, numBlocks, numThreadsPerBlock);

		_avgOut->add(_avgOutInc, 1, 1, numBlocks, numThreadsPerBlock);

		numBlocks = 1;
		_outBiasInc->add(_dE_db_j, _mom, -_epsHidBias / _minibatchSize, numBlocks, \
				numThreadsPerBlock);
		_outBiases->add(_outBiasInc, 1, 1, numBlocks, numThreadsPerBlock);
	}
	delete y_i_T;
}
































