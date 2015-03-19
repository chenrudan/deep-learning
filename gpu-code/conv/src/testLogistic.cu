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


int main(){

    pars* logistic = new pars;

    logistic->epsHidVis = 0.001;
    logistic->epsHidBias = 0.001;
    logistic->epsAvgOut = 0.13;
    logistic->epsOutBias = 0.13;
    logistic->mom = 0;
    logistic->wcHidVis = 0;
    logistic->wcAvgOut = 0;
    logistic->inSize = 28; 
    logistic->inChannel = 1;
    logistic->filterSize = 5;
    logistic->numFilters = 16; 
    logistic->numOut = 10; 
    logistic->trainNum = 50000;
    logistic->validNum = 10000;
    logistic->minibatchSize = 1000;
    logistic->numMinibatches = logistic->trainNum / logistic->minibatchSize;
    logistic->numValidBatches = logistic->validNum / logistic->minibatchSize;
    logistic->numEpoches = 100; 
    logistic->nPush = 1;
    logistic->nFetch = 1;

/*
	cout << "=========================\n" \
		 << "train: " << trainNum \
		 << "\nvalid: " << validNum \
		 << "\nfiltersize: " << filterSize \
		 << "\nnumFilters: " << numFilters \
		 << "\nepsHidVis: " << epsHidVis \
		 << "\nepsHidBias: " << epsHidBias \
		 << "\nepsAvgOut: " << epsAvgOut \
		 << "\nepsOutBias: " << epsOutBias \
		 << "\nmom: " << mom \
		 << "\nwcHidVis: " << wcHidVis \
		 << "\nwcAvgOut: " << wcAvgOut << endl;
*/	
	NVMatrix* nvTrainData = new NVMatrix(logistic->trainNum, logistic->inSize * logistic->inSize);
	NVMatrix* nvValidData = new NVMatrix(logistic->validNum, logistic->inSize * logistic->inSize);
	NVMatrix* nvTrainLabel = new NVMatrix(logistic->trainNum, 1);
	NVMatrix* nvValidLabel = new NVMatrix(logistic->validNum, 1);

	readData(nvTrainData, "../data/input/mnist_train.bin", true);
	readData(nvValidData, "../data/input/mnist_valid.bin", true);
	readData(nvTrainLabel, "../data/input/mnist_label_train.bin", false);
	readData(nvValidLabel, "../data/input/mnist_label_valid.bin", false);
	
	float* trainDataPtr = nvTrainData->getDevData();
	float* trainLabelPtr = nvTrainLabel->getDevData();
	float* validDataPtr = nvValidData->getDevData();
	float* validLabelPtr = nvValidLabel->getDevData();

	NVMatrix* miniTrainData = new NVMatrix(logistic->minibatchSize, logistic->inSize * logistic->inSize);
	NVMatrix* miniTrainLabel = new NVMatrix(logistic->minibatchSize, 1);
	NVMatrix* miniValidData = new NVMatrix(logistic->minibatchSize, logistic->inSize * logistic->inSize);
	NVMatrix* miniValidLabel = new NVMatrix(logistic->minibatchSize, 1);


	int lenPO = logistic->inSize * logistic->inSize;
	Matrix* hHidVis = new Matrix(logistic->numFilters, logistic->filterSize * logistic->filterSize);
	Matrix* hHidBiases = new Matrix(logistic->numFilters, 1);
	Matrix* hAvgout = new Matrix(lenPO, logistic->numOut);
	Matrix* hOutBiases = new Matrix(1, logistic->numOut);

	initW(hHidVis->getData(), logistic->numFilters * logistic->filterSize * logistic->filterSize);
	memset(hHidBiases->getData(), 0, sizeof(float) * logistic->numFilters);
	memset(hAvgout->getData(), 0, sizeof(float) * lenPO * logistic->numOut);
	memset(hOutBiases->getData(), 0, sizeof(float) * logistic->numOut);
	
//	readPars(hHidVis, "hHidVis_t1.bin");
//	readPars(hHidBiases, "hHidBiases_t1.bin");
//	readPars(hAvgout, "hAvgout_t1.bin");
//	readPars(hOutBiases, "hOutBiases_t1.bin");

	//ConvNet layer1(hHidVis, hAvgout, hHidBiases, hOutBiases, logistic);
	Logistic layer1(hAvgout, hOutBiases, logistic);
	layer1.initCuda();

	double loglihood = 0;
	int numError = 0;
	
	clock_t t;
	t = clock();

	for(int epochIdx = 0; epochIdx < logistic->numEpoches; epochIdx++){
		miniTrainData->setPtr(trainDataPtr);
		miniTrainLabel->setPtr(trainLabelPtr);
		miniValidData->setPtr(validDataPtr);
		miniValidLabel->setPtr(validLabelPtr);
//		cout << "\n--------train for epoch "<< epochIdx << "--------\n";
		for(int batchIdx = 0; batchIdx < logistic->numMinibatches; batchIdx++){
			//读取数据
			
			int error = 0;
			//Forward pass
			
			layer1.computeClassOutputs(miniTrainData);
            layer1.computeDerivs(miniTrainData, miniTrainLabel);
            layer1.updatePars();
            layer1.computeError(miniTrainLabel, error);
	//		layer1.computeLogistic(miniTrainData, miniTrainLabel, true);

	//		loglihood = layer1.computeError(miniTrainLabel, error);
			miniTrainData->changePtr(logistic->minibatchSize * logistic->inSize * logistic->inSize);
			miniTrainLabel->changePtr(logistic->minibatchSize);

			numError += error;

			if(batchIdx == logistic->numMinibatches - 1){
				int errorValid = 0;
				float loglihoodValid = 0;
				for(int validIdx = 0; validIdx < logistic->validNum / logistic->minibatchSize; validIdx++){

	//				layer1.computeLogistic(miniValidData, miniValidLabel, false);
	//				loglihoodValid = layer1.computeError(miniValidLabel, errorValid)		
					layer1.computeClassOutputs(miniValidData);
                    layer1.computeError(miniValidLabel, errorValid);

					miniValidData->changePtr(logistic->minibatchSize * logistic->inSize * logistic->inSize);
					miniValidLabel->changePtr(logistic->minibatchSize);
				}
//			cout << "--------valid for epoch "<< epochIdx << "--------\n";
			cout << "epoch: " << epochIdx 
//		<< ",total number: " << validNum 
//				<< ",error number: " << errorValid 
				<< ",error rate: " << (float)errorValid/logistic->validNum  << endl;
//				<< ",negitive likelihood: " << loglihoodValid/validNum << endl;
			}
		}
	}
	t = clock() - t;
	cout << "layer1 uses " << ((float)t/CLOCKS_PER_SEC)/logistic->numEpoches << " seconds. \n";
	savePars(hHidVis, "../data/pars/hHidVis_t1.bin");
	savePars(hHidBiases, "../data/pars/hHidBiases_t1.bin");
	savePars(hAvgout, "../data/pars/hAvgout_t1.bin");
	savePars(hOutBiases, "../data/pars/hOutBiases_t1.bin");
	
	delete nvTrainData;
	delete nvTrainLabel;
	delete nvValidData;
	delete nvValidLabel;
	delete miniTrainData;
	delete miniTrainLabel;
	delete miniValidData;
	delete miniValidLabel;

	return 0;
}
