/*
 * filename:testMultiGpu.cu
 */

#include <iostream>
#include <fstream>
#include <time.h>
#include <cmath>
#include "mpi.h"
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

int main(int argc, char** argv){

	int rank;
	int numProcess;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&numProcess);

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
	int minibatchSize = 1000;
	int numMinibatches = trainNum / (minibatchSize * numProcess);
	int numValidBatches = validNum / (minibatchSize * numProcess);
	int numEpoches = 300; 
	int inChannel = 1;


	float* trainDataPtr;
	float* trainLabelPtr;
	float* validDataPtr;
	float* validLabelPtr;

	int inSqrt = inSize * inSize;

	NVMatrix* nvTrainData = new NVMatrix(trainNum, inSize * inSize);
	NVMatrix* nvValidData = new NVMatrix(validNum, inSize * inSize);
	NVMatrix* nvTrainLabel = new NVMatrix(trainNum, 1);
	NVMatrix* nvValidLabel = new NVMatrix(validNum, 1);
	
	NVMatrix* miniTrainData = new NVMatrix(minibatchSize, inSqrt);
	NVMatrix* miniTrainLabel = new NVMatrix(minibatchSize, 1);
	NVMatrix* miniValidData = new NVMatrix(minibatchSize, inSqrt);
	NVMatrix* miniValidLabel = new NVMatrix(minibatchSize, 1);
	if(rank == 0){
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
			<< "\nwcAvgOut: " << wcAvgOut \
			<< "\n========================" << endl;

		//0号进程来读取输入数据
		readData(nvTrainData, "../data/input/mnist_train.bin", true);
		readData(nvValidData, "../data/input/mnist_valid.bin", true);
		readData(nvTrainLabel, "../data/input/mnist_label_train.bin", false);
		readData(nvValidLabel, "../data/input/mnist_label_valid.bin", false);

		//0号进程移动数据指针
		trainDataPtr = nvTrainData->getDevData();
		trainLabelPtr = nvTrainLabel->getDevData();
		validDataPtr = nvValidData->getDevData();
		validLabelPtr = nvValidLabel->getDevData();
	}

	//参数全部都需要
	int hidVisLen = numFilters * filterSize * filterSize;
	int hidBiasLen = numFilters;
	int avgOutLen = inSqrt * numOut;
	int outBiasLen = numOut;
	Matrix* hHidVis = new Matrix(numFilters, filterSize * filterSize);
	Matrix* hHidBiases = new Matrix(numFilters, 1);
	Matrix* hAvgout = new Matrix(inSqrt, numOut);
	Matrix* hOutBiases = new Matrix(1, numOut);

	int miniDataLen = minibatchSize * inSqrt;
	int miniLabelLen = minibatchSize;
	//0号进程初始化参数，进行分发
	if(rank == 0){
		initW(hHidVis->getData(), hidVisLen);
		memset(hHidBiases->getData(), 0, sizeof(float) * numFilters);
		memset(hAvgout->getData(), 0, sizeof(float) * avgOutLen);
		memset(hOutBiases->getData(), 0, sizeof(float) * numOut);
		//	readPars(hHidVis, "hHidVis_t1.bin");
		//	readPars(hHidBiases, "hHidBiases_t1.bin");
		//	readPars(hAvgout, "hAvgout_t1.bin");
		//	readPars(hOutBiases, "hOutBiases_t1.bin");
	}
	//先只处理一层的logistic
	NVMatrix* avgOut, *outBiases;
	MPI_Bcast(hAvgout->getData(), avgOutLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(hOutBiases->getData(), outBiasLen, MPI_FLOAT, 0, MPI_COMM_WORLD);

	cudaSetDevice(0);

	ConvNet layer1(hHidVis, hAvgout, hHidBiases, hOutBiases, epsHidVis, epsAvgOut, \
			epsHidBias, epsOutBias, mom, wcHidVis, wcAvgOut, minibatchSize, \
			inSize, filterSize, inChannel, numFilters);
	layer1.initCuda();

	double loglihood = 0;
	int numError = 0;

	int nPush = 1;
	int nFetch = 1;

	clock_t t;
	if(rank == 0)
		t = clock();
	for(int epochIdx = 0; epochIdx < numEpoches; epochIdx++){
		if(rank == 0){
			nvTrainData->setPtr(trainDataPtr);
			nvTrainLabel->setPtr(trainLabelPtr);
			nvValidData->setPtr(validDataPtr);
			nvValidLabel->setPtr(validLabelPtr);
		}
		for(int batchIdx = 0; batchIdx < numMinibatches; batchIdx++){
			//读取数据
			MPI_Scatter(nvTrainData, miniDataLen, MPI_FLOAT, miniTrainData, \
					miniDataLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
			//MPI_Scatter(nvTrainLabel, miniLabelLen, MPI_FLOAT, miniTrainLabel, \
					miniLabelLen, MPI_FLOAT, 0, MPI_COMM_WORLD);

		/*
			int error = 0;
			//Forward pass
			cout << "done1\n";
			layer1.computeLogistic(miniTrainData, miniTrainLabel, true);

			cout << "done2\n";
			loglihood = layer1.computeError(miniTrainLabel, error);
			cout << "done3\n";
			if(rank == 0){
				nvTrainData->changePtr(numProcess * miniDataLen);
				nvTrainLabel->changePtr(numProcess * miniLabelLen);
			}
			//gather参数，然后scatter
			avgOut = layer1.getAvgOut();
			outBiases = layer1.getOutBias();
			if((batchIdx + 1) % nPush == 0){
				MPI_Gather(rank*avgOutLen/numProcess + avgOut->getDevData(), \
						avgOutLen/numProcess, MPI_FLOAT, avgOut->getDevData(), \
						avgOutLen/numProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
				MPI_Gather(rank*outBiasLen/numProcess + outBiases->getDevData(), \
						outBiasLen/numProcess, MPI_FLOAT, outBiases->getDevData(), \
						outBiasLen/numProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);
			}
			if((batchIdx + 1) % nFetch == 0){
				MPI_Bcast(hAvgout->getData(), avgOutLen, MPI_FLOAT, \
						0, MPI_COMM_WORLD);
				MPI_Bcast(hOutBiases->getData(), outBiasLen, MPI_FLOAT, \
						0, MPI_COMM_WORLD);
			}



			if(batchIdx == numMinibatches - 1){
				int errorValid = 0;
				float loglihoodValid = 0;
				for(int validIdx = 0; validIdx < numValidBatches; validIdx++){
					MPI_Scatter(nvTrainData, miniDataLen, MPI_FLOAT, miniTrainData, \
							miniDataLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
					MPI_Scatter(nvTrainLabel, miniLabelLen, MPI_FLOAT, \
							miniTrainLabel, miniLabelLen, MPI_FLOAT, \
							0, MPI_COMM_WORLD);

					layer1.computeLogistic(miniValidData, miniValidLabel, false);
					loglihoodValid = layer1.computeError(miniValidLabel, errorValid);

					if(rank == 0){
						nvValidData->changePtr(numProcess * miniDataLen);
						nvValidLabel->changePtr(numProcess * miniLabelLen);
					}
				}
				MPI_Reduce(&errorValid, &errorValid, 1, MPI_INT, MPI_SUM, \
						0, MPI_COMM_WORLD);
				if(rank == 0){
					t = clock() - t;
					cout << " " << (float)t/CLOCKS_PER_SEC << " seconds. \n";
					t = clock();
					cout << "epoch: " << epochIdx 
						<< ",error rate: " << (float)errorValid/validNum  << endl;
				}
			}
			*/
		}
	}
	if(rank == 0){
		savePars(hHidVis, "../data/pars/hHidVis_t1.bin");
		savePars(hHidBiases, "../data/pars/hHidBiases_t1.bin");
		savePars(hAvgout, "../data/pars/hAvgout_t1.bin");
		savePars(hOutBiases, "../data/pars/hOutBiases_t1.bin");
	}
	delete nvTrainData;
	delete nvTrainLabel;
	delete nvValidData;
	delete nvValidLabel;
	delete miniTrainData;
	delete miniTrainLabel;
	delete miniValidData;
	delete miniValidLabel;

	MPI_Finalize();
	return 0;
}
