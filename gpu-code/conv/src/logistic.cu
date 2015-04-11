/*
 * filename: logistic.cu
 */
#include <time.h>
#include "logistic.cuh"
#include "logistic_kernel.cuh"

using namespace std;

Logistic::Logistic(pars* netWork){
	this->_numIn				 = netWork->numIn;
	this->_numOut                = netWork->numOut;

	//w_hk的learning rate
	this->_epsAvgOut             = netWork->epsAvgOut;
	//out bias learning rate
	this->_epsOutBias            = netWork->epsOutBias;
	//上一次更新的参数控制增长趋势
	this->_mom                   = netWork->mom;
	//hidden原值的参数
	this->_wcHidVis              = netWork->wcHidVis;
	//out原值的参数
	this->_wcAvgOut              = netWork->wcAvgOut;

	this->_minibatchSize         = netWork->minibatchSize;
	this->_finePars				 = netWork->finePars;

	cublasCreate(&handle);
}

Logistic::~Logistic() {

	delete _avgOut;
	delete _avgOutInc;
	delete _outBiases;
	delete _outBiasInc;

	delete  _y_j;
	delete  _dE_dy_j;
	delete _dE_db_j;
	delete _dE_dw_ij;
	cublasDestroy(handle);
}

void Logistic::initCuda() {

	this->_avgOut            = new NVMatrix(_numIn, _numOut);
//					NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	this->_outBiases         = new NVMatrix(1, _numOut);
//					NVMatrix::ALLOC_ON_UNIFIED_MEMORY);

	this->_y_j               = new NVMatrix(_minibatchSize, _numOut);

	this->_dE_dy_j           = new NVMatrix(_y_j);
	this->_dE_db_j           = new NVMatrix(_outBiases);
	this->_dE_dw_ij          = new NVMatrix(_avgOut);

	this->_avgOutInc         = new NVMatrix(_avgOut);
	this->_outBiasInc        = new NVMatrix(1, _numOut);
	this->_avgOutInc->zeros();
	this->_outBiasInc->zeros();
}

void Logistic::computeClassOutputs(NVMatrix* miniData){
//miniData->showValue("data");
	miniData->rightMult(_avgOut, 1, _y_j, handle);
	_y_j->addRowVector(_outBiases);
	_y_j->apply(NVMatrix::SOFTMAX);

//_y_j->showValue("yj1");
}

double Logistic::computeError(const NVMatrix* const miniLabels, int& numError){

	Matrix* hlabels = new Matrix(miniLabels->getNumRows(), miniLabels->getNumCols());
	miniLabels->copyToHost(hlabels);
	Matrix* y_j_CPU = new Matrix(_y_j->getNumRows(), _y_j->getNumCols());
	_y_j->copyToHost(y_j_CPU);
	Matrix* correctProbs = new Matrix(_y_j->getNumRows(), 1);
	NVMatrix* maxPosOfOutGpu = new NVMatrix(_y_j->getNumRows(), 1);
	_y_j->maxPosInRow(maxPosOfOutGpu);
	Matrix* maxPosCpu = new Matrix(_y_j->getNumRows(), 1);
	maxPosOfOutGpu->copyToHost(maxPosCpu);
	for (int c = 0; c < _y_j->getNumRows(); c++) {
		int trueLabel = hlabels->getCell(c, 0);
		int predictLabel = maxPosCpu->getCell(c, 0);
		correctProbs->getCell(c, 0) = y_j_CPU->getCell(c, trueLabel);

//cout << predictLabel << ":" << trueLabel << " ";
		if(predictLabel != trueLabel)
			numError++;
	}
//cout << endl;
	correctProbs->apply(Matrix::LOG);
	double result = -correctProbs->sum();
	cudaThreadSynchronize();

	delete hlabels;
	delete y_j_CPU;
	delete correctProbs;
	delete maxPosOfOutGpu;
	delete maxPosCpu;
	return result;
}

void Logistic::computeDerivs(NVMatrix* miniData, NVMatrix* miniLabels, NVMatrix* dE_dy_i){
	assert(miniLabels->getNumRows() == miniData->getNumRows());

	const int numThreads = DIVUP(_numOut, ADD_BLOCK_SIZE) * ADD_BLOCK_SIZE;
	compute_dE_dy<<<_minibatchSize, numThreads>>>(_y_j->getDevData(), \
			miniLabels->getDevData(), _dE_dy_j->getDevData(), _numOut);

	NVMatrix* data_T = new NVMatrix(miniData->getNumCols(), miniData->getNumRows());
	miniData->getTranspose(data_T);

	data_T->rightMult(_dE_dy_j, 1, _dE_dw_ij, handle);
	_dE_dy_j->sumRow(_dE_db_j);

	NVMatrix* avgOut_T = new NVMatrix(_avgOut->getNumCols(), _avgOut->getNumRows());
    _avgOut->getTranspose(avgOut_T);
    _dE_dy_j->rightMult(avgOut_T, 1, dE_dy_i, handle);

	delete avgOut_T;
	delete data_T;
}

void Logistic::updatePars(){
	_avgOutInc->addSum(_avgOut, _dE_dw_ij, _mom, -_wcAvgOut, \
			-_epsAvgOut / _minibatchSize);
	_avgOut->add(_avgOutInc, 1, 1);

	_outBiasInc->add(_dE_db_j, _mom, -_epsOutBias / _minibatchSize);
	_outBiases->add(_outBiasInc, 1, 1);
}



