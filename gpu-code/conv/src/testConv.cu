/*
 * filename:main.cu
 */

#include <iostream>
#include <fstream>
#include <time.h>
#include <cmath>

#include "matrix.h"
#include "nvmatrix.cuh"
#include "convnet.cuh"
#include "convnet_kernel.cuh"
#include "utils.h"
#include "logistic.cuh"

using namespace std;

int numProcess;
int rank;


int main(){
	pars* cnn = new pars;
	pars* logistic = new pars;

    cnn->epsHidVis = 0.1;
   	cnn->epsHidBias = 0.1;
    cnn->mom = 0;
    cnn->wcHidVis = 0;
    cnn->inSize = 28; 
    cnn->inChannel = 1;
    cnn->filterSize = 5;
    cnn->numFilters = 16; 
    cnn->trainNum = 50000;
    cnn->validNum = 10000;
    cnn->minibatchSize = 1000;
    cnn->numMinibatches = cnn->trainNum / cnn->minibatchSize;
    cnn->numValidBatches = cnn->validNum / cnn->minibatchSize;
    cnn->numEpoches = 20; 
    cnn->nPush = 1;
    cnn->nFetch = 1;

    logistic->wcAvgOut = 0;
    logistic->epsAvgOut = 0.1;
    logistic->epsOutBias = 0.1;
    logistic->mom = 0;
    logistic->numOut = 10; 
	logistic->minibatchSize = 1000;

	clock_t t;
	t = clock();

	cout << "=========================\n" \
		 << "train: " << cnn->trainNum \
		 << "\nvalid: " << cnn->validNum \
		 << "\nfiltersize: " << cnn->filterSize \
		 << "\nnumFilters: " << cnn->numFilters \
		 << "\nepsHidVis: " << cnn->epsHidVis \
		 << "\nepsHidBias: " << cnn->epsHidBias \
		 << "\nepsAvgOut: " << cnn->epsAvgOut \
		 << "\nepsOutBias: " << cnn->epsOutBias \
		 << "\nmom: " << cnn->mom \
		 << "\nwcHidVis: " << cnn->wcHidVis \
		 << "\nwcAvgOut: " << cnn->wcAvgOut << endl;
	
	NVMatrix* nvTrainData = new NVMatrix(cnn->trainNum, cnn->inSize * cnn->inSize);
	NVMatrix* nvValidData = new NVMatrix(cnn->validNum, cnn->inSize * cnn->inSize);
	NVMatrix* nvTrainLabel = new NVMatrix(cnn->trainNum, 1);
	NVMatrix* nvValidLabel = new NVMatrix(cnn->validNum, 1);

	readData(nvTrainData, "../data/input/mnist_train.bin", true);
	readData(nvValidData, "../data/input/mnist_valid.bin", true);
	readData(nvTrainLabel, "../data/input/mnist_label_train.bin", false);
	readData(nvValidLabel, "../data/input/mnist_label_valid.bin", false);
	
	float* trainDataPtr = nvTrainData->getDevData();
	float* trainLabelPtr = nvTrainLabel->getDevData();
	float* validDataPtr = nvValidData->getDevData();
	float* validLabelPtr = nvValidLabel->getDevData();

	NVMatrix* miniTrainData = new NVMatrix(cnn->minibatchSize, cnn->inSize * cnn->inSize);
	NVMatrix* miniTrainLabel = new NVMatrix(cnn->minibatchSize, 1);
	NVMatrix* miniValidData = new NVMatrix(cnn->minibatchSize, cnn->inSize * cnn->inSize);
	NVMatrix* miniValidLabel = new NVMatrix(cnn->minibatchSize, 1);


	int lenPO = cnn->numFilters * POOL_FORWARD_SIZE * POOL_FORWARD_SIZE;
	Matrix* hHidVis = new Matrix(cnn->numFilters, cnn->filterSize * cnn->filterSize);
	Matrix* hHidBiases = new Matrix(cnn->numFilters, 1);
	Matrix* hAvgout = new Matrix(lenPO, logistic->numOut);
	Matrix* hOutBiases = new Matrix(1, logistic->numOut);

	initW(hHidVis->getData(), cnn->numFilters * cnn->filterSize * cnn->filterSize);
	memset(hHidBiases->getData(), 0, sizeof(float) * cnn->numFilters);
	memset(hAvgout->getData(), 0, sizeof(float) * lenPO * logistic->numOut);
	memset(hOutBiases->getData(), 0, sizeof(float) * logistic->numOut);
	
//	readPars(hHidVis, "hHidVis_t1.bin");
//	readPars(hHidBiases, "hHidBiases_t1.bin");
//	readPars(hAvgout, "hAvgout_t1.bin");
//	readPars(hOutBiases, "hOutBiases_t1.bin");

	ConvNet layer1(hHidVis, hHidBiases, cnn);
	layer1.initCuda();
	Logistic layer2(hAvgout, hOutBiases, logistic);
	layer2.initCuda();

	NVMatrix* avgOut;
	NVMatrix* dE_dy_j;
	NVMatrix* y_i;

	double loglihood = 0;
	int numError = 0;

	for(int epochIdx = 0; epochIdx < cnn->numEpoches; epochIdx++){
		miniTrainData->setPtr(trainDataPtr);
		miniTrainLabel->setPtr(trainLabelPtr);
		miniValidData->setPtr(validDataPtr);
		miniValidLabel->setPtr(validLabelPtr);
		for(int batchIdx = 0; batchIdx < cnn->numMinibatches; batchIdx++){
			//读取数据
			int error = 0;
			//Forward pass
			layer1.computeConvOutputs(miniTrainData);
			layer1.computeMaxOutputs();
			y_i = layer1.getYI();
			layer2.computeClassOutputs(y_i);
			loglihood = layer2.computeError(miniTrainLabel, error);
			dE_dy_j = layer2.getDEDYJ();
			avgOut = layer2.getAvgOut();
			layer2.computeDerivs(y_i, miniTrainLabel);
			layer1.computeDerivs(miniTrainData, dE_dy_j, avgOut);
			layer1.updatePars();
			layer2.updatePars();
			
			miniTrainData->changePtr(cnn->minibatchSize * cnn->inSize * cnn->inSize * cnn->inChannel);
			miniTrainLabel->changePtr(cnn->minibatchSize);
			numError += error;
			if(batchIdx == cnn->numMinibatches - 1){
				int errorValid = 0;
				float loglihoodValid = 0;
				for(int validIdx = 0; validIdx < cnn->validNum / cnn->minibatchSize; \
							validIdx++){

					layer1.computeConvOutputs(miniValidData);
					layer1.computeMaxOutputs();
					y_i = layer1.getYI();
					layer2.computeClassOutputs(y_i);
					loglihoodValid = layer2.computeError(miniValidLabel, errorValid);
					
					miniValidData->changePtr(cnn->minibatchSize * cnn->inSize * cnn->inSize * cnn->inChannel);
					miniValidLabel->changePtr(cnn->minibatchSize);
			}
		//	layer1.transfarLowerAvgOut();
			cout << "--------valid for epoch "<< epochIdx << "--------\n";
			cout << "epoch: " << epochIdx \
				<< ",error rate: " << (float)errorValid/cnn->validNum  \
				<< ",negitive likelihood: " << loglihoodValid/cnn->validNum << endl;
			}
		}
		t = clock() - t;
		cout << "epoch: " << (float)t/CLOCKS_PER_SEC << " seconds. \n";
		t = clock();
	}

//	savePars(hHidVis, "../data/pars/hHidVis_t1.bin");
//	savePars(hHidBiases, "../data/pars/hHidBiases_t1.bin");
//	savePars(hAvgout, "../data/pars/hAvgout_t1.bin");
//	savePars(hOutBiases, "../data/pars/hOutBiases_t1.bin");
	
	delete nvTrainData;
	delete nvTrainLabel;
	delete nvValidData;
	delete nvValidLabel;
	delete miniTrainData;
	delete miniTrainLabel;
	delete miniValidData;
	delete miniValidLabel;

	delete cnn;
	return 0;
}










