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

using namespace std;

void initW(float* a, int length){
	srand((unsigned)time(NULL));
	float bound = sqrt(1.0 / length);
	for(int i = 0; i < length; i++){
		int k = rand() % 200;
		if(k < 100)
			a[i] = (k/100.0)*(-bound);
		else
			a[i] = ((k - 100)/100.0)*bound; 
	}
}

void readPars(Matrix* par, string filename){
	ifstream fin1(filename.c_str(), ios::binary);
	int dataLen = par->getNumRows() * par->getNumCols();
	fin1.read((char*)(par->getData()), sizeof(float) * dataLen);
	fin1.close();
}

void savePars(Matrix* par, string filename){
	ofstream fout(filename.c_str(), ios::binary);
	int dataLen = par->getNumRows() * par->getNumCols();
	fout.write((char*)(par->getData()), sizeof(float) * dataLen);
	fout.close();
}

void readData(NVMatrix* nvData, string filename, bool isData){
	int length = nvData->getNumRows() * nvData->getNumCols();
	ifstream fin(filename.c_str(), ios::binary);
	float* data = new float[length];
	char* readData = new char[length];
	fin.read(readData, length);
	for(int i = 0; i < length; i++){
		unsigned char tmp = readData[i];
		if(isData){
			data[i] = (int)tmp / 255.0;
		}
		else
			data[i] = (int)tmp;
	}
	nvData->copyFromHost(data, length);
	fin.close();
}

int main(){

	float epsHidVis = 0.001;
	float epsHidBias = 0.001;
	float epsAvgOut = 0.13;
	float epsOutBias = 0.13;
	float mom = 0;
	float wcHidVis = 0;
	float wcAvgOut = 0;

	int inSize = 28;
	int filterSize = 5;
	int numFilters = 16;
	int numOut = 10;
	int trainNum = 50000;
	int validNum = 10000;
	int minibatchSize = 1024;
	int numMinibatches = trainNum / minibatchSize;
	int numEpoches = 300; 
	int inChannel = 1;
	

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
	
	NVMatrix* nvTrainData = new NVMatrix(trainNum, inSize * inSize);
	NVMatrix* nvValidData = new NVMatrix(validNum, inSize * inSize);
	NVMatrix* nvTrainLabel = new NVMatrix(trainNum, 1);
	NVMatrix* nvValidLabel = new NVMatrix(validNum, 1);

	readData(nvTrainData, "../data/input/mnist_train.bin", true);
	readData(nvValidData, "../data/input/mnist_valid.bin", true);
	readData(nvTrainLabel, "../data/input/mnist_label_train.bin", false);
	readData(nvValidLabel, "../data/input/mnist_label_valid.bin", false);
	
	float* trainDataPtr = nvTrainData->getDevData();
	float* trainLabelPtr = nvTrainLabel->getDevData();
	float* validDataPtr = nvValidData->getDevData();
	float* validLabelPtr = nvValidLabel->getDevData();

	NVMatrix* miniTrainData = new NVMatrix(minibatchSize, inSize * inSize);
	NVMatrix* miniTrainLabel = new NVMatrix(minibatchSize, 1);
	NVMatrix* miniValidData = new NVMatrix(minibatchSize, inSize * inSize);
	NVMatrix* miniValidLabel = new NVMatrix(minibatchSize, 1);


	int lenPO = inSize * inSize;
	Matrix* hHidVis = new Matrix(numFilters, filterSize * filterSize);
	Matrix* hHidBiases = new Matrix(numFilters, 1);
	Matrix* hAvgout = new Matrix(lenPO, numOut);
	Matrix* hOutBiases = new Matrix(1, numOut);

	initW(hHidVis->getData(), numFilters * filterSize * filterSize);
	memset(hHidBiases->getData(), 0, sizeof(float) * numFilters);
	memset(hAvgout->getData(), 0, sizeof(float) * lenPO * numOut);
	memset(hOutBiases->getData(), 0, sizeof(float) * numOut);
	
//	readPars(hHidVis, "hHidVis_t1.bin");
//	readPars(hHidBiases, "hHidBiases_t1.bin");
//	readPars(hAvgout, "hAvgout_t1.bin");
//	readPars(hOutBiases, "hOutBiases_t1.bin");

	ConvNet layer1(hHidVis, hAvgout, hHidBiases, hOutBiases, epsHidVis, epsAvgOut, \
			epsHidBias, epsOutBias, mom, wcHidVis, wcAvgOut, minibatchSize, \
			inSize, filterSize, inChannel, numFilters);
	layer1.initCuda();

	double loglihood = 0;
	int numError = 0;
	
	clock_t t;
	t = clock();

	for(int epochIdx = 0; epochIdx < numEpoches; epochIdx++){
		miniTrainData->setPtr(trainDataPtr);
		miniTrainLabel->setPtr(trainLabelPtr);
		miniValidData->setPtr(validDataPtr);
		miniValidLabel->setPtr(validLabelPtr);
//		cout << "\n--------train for epoch "<< epochIdx << "--------\n";
		for(int batchIdx = 0; batchIdx < numMinibatches; batchIdx++){
			//读取数据
			
			int error = 0;
			//Forward pass
			layer1.computeLogistic(miniTrainData, miniTrainLabel, true);

			loglihood = layer1.computeError(miniTrainLabel, error);
			miniTrainData->changePtr(minibatchSize * inSize * inSize);
			miniTrainLabel->changePtr(minibatchSize);

			numError += error;

			if(batchIdx == numMinibatches - 1){
				int errorValid = 0;
				float loglihoodValid = 0;
				for(int validIdx = 0; validIdx < validNum / minibatchSize; validIdx++){

					layer1.computeLogistic(miniValidData, miniValidLabel, false);
					loglihoodValid = layer1.computeError(miniValidLabel, errorValid);
					
					miniValidData->changePtr(minibatchSize * inSize * inSize);
					miniValidLabel->changePtr(minibatchSize);
				}
	t = clock() - t;
	cout << " " << (float)t/CLOCKS_PER_SEC << " seconds. \n";
	t = clock();
//			cout << "--------valid for epoch "<< epochIdx << "--------\n";
			cout << "epoch: " << epochIdx 
//		<< ",total number: " << validNum 
//				<< ",error number: " << errorValid 
				<< ",error rate: " << (float)errorValid/validNum  << endl;
//				<< ",negitive likelihood: " << loglihoodValid/validNum << endl;
			}
		}
	}
//	t = clock() - t;
//	cout << "layer1 uses " << (float)t/CLOCKS_PER_SEC << " seconds. \n";
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
