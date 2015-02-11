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
	//float sum = 0;
	for(int i = 0; i < length; i++){
		unsigned char tmp = readData[i];
		if(isData){
			data[i] = (int)tmp / 255.0;
	//		sum += data[i];
		}
		else
			data[i] = (int)tmp;
	}
/*	if(isData){
		sum /= length;
		for(int i = 0; i < length; i++){
			data[i] = data[i] - sum;
		}	
	}
*/
	nvData->copyFromHost(data, length);
	fin.close();
}

int main(){

	float epsHidVis = 0.1;
	float epsHidBias = 0.1;
	float epsAvgOut = 0.1;
	float epsOutBias = 0.1;
	float mom = 0;
	float wcHidVis = 0.00001;
	float wcAvgOut = 0.00001;

	int inSize = 28;
	int filterSize = 5;
	int numFilters = 16;
	int numOut = 10;
	int trainNum = 50000;
	int validNum = 10000;
	int minibatchSize = 16;
	int numMinibatches = trainNum / minibatchSize;
	int numEpoches = 200; 
	int inChannel = 1;
	
	clock_t t;
	t = clock();

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

	readData(nvTrainData, "mnist_train.bin", true);
	readData(nvValidData, "mnist_valid.bin", true);
	readData(nvTrainLabel, "mnist_label_train.bin", false);
	readData(nvValidLabel, "mnist_label_valid.bin", false);
	
	float* trainDataPtr = nvTrainData->getDevData();
	float* trainLabelPtr = nvTrainLabel->getDevData();
	float* validDataPtr = nvValidData->getDevData();
	float* validLabelPtr = nvValidLabel->getDevData();

	NVMatrix* miniTrainData = new NVMatrix(minibatchSize, inSize * inSize);
	NVMatrix* miniTrainLabel = new NVMatrix(minibatchSize, 1);
	NVMatrix* miniValidData = new NVMatrix(minibatchSize, inSize * inSize);
	NVMatrix* miniValidLabel = new NVMatrix(minibatchSize, 1);


	int lenPO = numFilters * POOL_FORWARD_SIZE * POOL_FORWARD_SIZE;
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
//		t = clock();
			layer1.computeConvOutputs(miniTrainData);
//		t = clock() - t;
//		cout << "conv: " << (float)t/CLOCKS_PER_SEC << " seconds. \n";
//		t = clock();
			layer1.computeMaxOutputs();
			layer1.computeClassOutputs();
//		t = clock() - t;
//		cout << "classout: " << (float)t/CLOCKS_PER_SEC << " seconds. \n";
//		t = clock();
			loglihood = layer1.computeError(miniTrainLabel, error);
			layer1.computeDerivs(miniTrainData, miniTrainLabel);
//		t = clock() - t;
//		cout << "derivs: " << (float)t/CLOCKS_PER_SEC << " seconds. \n";
//		t = clock();
			layer1.updatePars();
			
			miniTrainData->changePtr(minibatchSize * inSize * inSize);
			miniTrainLabel->changePtr(minibatchSize);
			numError += error;
//		t = clock() - t;
//		cout << "update: " << (float)t/CLOCKS_PER_SEC << " seconds. \n";
//		t = clock();

//			cout << "epoch: " << epochIdx << ", minibatch: " << batchIdx \
				<< ",total number: " <<  minibatchSize \
				<< ",error number: " << error \
				<< ",error rate: " << (float)error/minibatchSize  \
				<< ",negitive likelihood: " << loglihood/minibatchSize << "\n\n";
			if(batchIdx == numMinibatches - 1){
				int errorValid = 0;
				float loglihoodValid = 0;
				for(int validIdx = 0; validIdx < validNum / minibatchSize; validIdx++){

					layer1.computeConvOutputs(miniValidData);
					layer1.computeMaxOutputs();
					layer1.computeClassOutputs();
					loglihoodValid += layer1.computeError(miniValidLabel, errorValid);
					
					miniValidData->changePtr(minibatchSize * inSize * inSize);
					miniValidLabel->changePtr(minibatchSize);
				}
			layer1.transfarLowerAvgOut();
			cout << "--------valid for epoch "<< epochIdx << "--------\n";
			cout << "epoch: " << epochIdx << ",total number: " << validNum \
				<< ",error number: " << errorValid \
				<< ",error rate: " << (float)errorValid/validNum  \
				<< ",negitive likelihood: " << loglihoodValid/validNum << endl;
			}
		}
		t = clock() - t;
		cout << "epoch: " << (float)t/CLOCKS_PER_SEC << " seconds. \n";
		t = clock();
	}

	savePars(hHidVis, "hHidVis_t1.bin");
	savePars(hHidBiases, "hHidBiases_t1.bin");
	savePars(hAvgout, "hAvgout_t1.bin");
	savePars(hOutBiases, "hOutBiases_t1.bin");
	
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
