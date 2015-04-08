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

void readData(NVMatrix* nvData, string filename, bool isData, int addZerosInFront = 0){
	int length = nvData->getNumRows() * nvData->getNumCols();
	ifstream fin(filename.c_str(), ios::binary);
	float* data = new float[length];
	char* readData = new char[length];
	fin.read(readData + addZerosInFront, length - addZerosInFront);
	for(int i = 0; i < length; i++){
		if(i < addZerosInFront)
			readData[i] = 0;
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
	MPI_Status status;
	MPI_Request req;

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
	int numMinibatches = trainNum / (minibatchSize * (numProcess - 1));
	int numValidBatches = validNum / (minibatchSize * (numProcess - 1));
	int numEpoches = 1000; 
	int inChannel = 1;


	float* trainDataPtr;
	float* trainLabelPtr;
	float* validDataPtr;
	float* validLabelPtr;

	int inSqrt = inSize * inSize;

	int hidVisLen = numFilters * filterSize * filterSize;
	//	int hidBiasLen = numFilters;
	int avgOutLen = inSqrt * numOut;
	int outBiasLen = numOut;

	int miniDataLen = minibatchSize * inSqrt;
	int miniLabelLen = minibatchSize;

	cudaSetDevice(rank%2);

	NVMatrix* nvTrainData;
	NVMatrix* nvValidData;
	NVMatrix* nvTrainLabel;
	NVMatrix* nvValidLabel;

	NVMatrix* avgOut;
	NVMatrix* outBiases;


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

		/*
		 *注意此处，一个线程作为server时，要在文件前面补上一个minibatch的空数据
		 */
		nvTrainData = new NVMatrix(trainNum, inSqrt, 
				NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
		nvValidData = new NVMatrix(validNum, inSqrt, \
				NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
		nvTrainLabel = new NVMatrix(trainNum, 1, \
				NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
		nvValidLabel = new NVMatrix(validNum, 1, \
				NVMatrix::ALLOC_ON_UNIFIED_MEMORY);

		avgOut = new NVMatrix(inSqrt, numOut, \
				NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
		outBiases = new NVMatrix(1, numOut, NVMatrix::ALLOC_ON_UNIFIED_MEMORY);

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
	}else{
		nvTrainData = new NVMatrix(minibatchSize, inSqrt, \
				NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
		nvValidData = new NVMatrix(minibatchSize, inSqrt, \
				NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
		nvTrainLabel = new NVMatrix(minibatchSize, 1, \
				NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
		nvValidLabel = new NVMatrix(minibatchSize, 1, \
				NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	}
	//参数全部都需要

	Matrix* hHidVis = new Matrix(numFilters, filterSize * filterSize);
	Matrix* hHidBiases = new Matrix(numFilters, 1);
	Matrix* hAvgout = new Matrix(inSqrt, numOut);
	Matrix* hOutBiases = new Matrix(1, numOut);

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

	MPI_Bcast(hAvgout->getData(), avgOutLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(hOutBiases->getData(), outBiasLen, MPI_FLOAT, 0, MPI_COMM_WORLD);

	ConvNet layer1(hHidVis, hAvgout, hHidBiases, hOutBiases, epsHidVis, epsAvgOut, \
			epsHidBias, epsOutBias, mom, wcHidVis, wcAvgOut, minibatchSize, \
			inSize, filterSize, inChannel, numFilters);
	if(rank != 0){
		layer1.initCuda();
	}
	double loglihood = 0;

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
			//使用send和receive
			if(rank == 0){
				for(int i = 1; i < numProcess; i++){
					MPI_Isend(nvTrainData->getDevData() + (i-1)*miniDataLen, \
							miniDataLen, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &req);
					MPI_Isend(nvTrainLabel->getDevData() + (i-1)*miniLabelLen, \
							miniLabelLen, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &req);
				}
			}
			else{
				MPI_Irecv(nvTrainData->getDevData(), miniDataLen, MPI_FLOAT, 0, \
						0, MPI_COMM_WORLD, &req);
				MPI_Irecv(nvTrainLabel->getDevData(), miniLabelLen, MPI_FLOAT, 0, \
						0, MPI_COMM_WORLD, &req);
			}

			int error = 0;
			if(rank != 0){
				//Forward pass

				layer1.computeLogistic(nvTrainData, nvTrainLabel, true);
				loglihood = layer1.computeError(nvTrainLabel, error);
			}

			if(rank == 0){
				nvTrainData->changePtr((numProcess-1) * miniDataLen);
				nvTrainLabel->changePtr((numProcess-1) * miniLabelLen);
			}

			//点对点的send，然后再recv
			if(rank != 0){
				avgOut = layer1.getAvgOut();
				outBiases = layer1.getOutBias();
			}
			if((batchIdx + 1) % nPush == 0){
				if(rank != 0){
					MPI_Isend((rank-1)*avgOutLen/(numProcess-1) \
							+ avgOut->getDevData(), avgOutLen/(numProcess-1), \
							MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &req);
					MPI_Isend((rank-1)*outBiasLen/(numProcess-1) \
							+ outBiases->getDevData(), outBiasLen/(numProcess-1), \
							MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &req);

				}else{
					for(int sender = 1; sender < numProcess; sender++){
						MPI_Irecv((sender-1)*avgOutLen/(numProcess-1) \
								+ avgOut->getDevData(), avgOutLen/(numProcess-1), \
								MPI_FLOAT, sender, 0, MPI_COMM_WORLD, \
								&req);
						MPI_Irecv((sender-1)*outBiasLen/(numProcess-1) \
								+ outBiases->getDevData(), outBiasLen/(numProcess-1), \
								MPI_FLOAT, sender, 0, MPI_COMM_WORLD, \
								&req);
					}
				}
			}
			if((batchIdx + 1) % nFetch == 0){
				if(rank == 0){
					for(int i = 1; i < numProcess; i++){
						MPI_Isend(avgOut->getDevData(), avgOutLen, MPI_FLOAT, i, \
								0, MPI_COMM_WORLD, &req);
						MPI_Isend(outBiases->getDevData(), outBiasLen, MPI_FLOAT, \
								i, 0, MPI_COMM_WORLD, &req);
					}
				}else{
					MPI_Irecv(avgOut->getDevData(), avgOutLen, MPI_FLOAT, 0, 0, \
							MPI_COMM_WORLD, &req);
					MPI_Irecv(outBiases->getDevData(), outBiasLen, MPI_FLOAT, \
							0, 0, MPI_COMM_WORLD, &req);
				}
			}
			/*	if(rank == 1){
				cout << "batchIdx: " << batchIdx << ",error: " \
				<< (float)error/minibatchSize \
				<< ",likelihood: "<< loglihood<< endl;
				}*/
			if(batchIdx == numMinibatches - 1){
				int errorValid = 0;
				float loglihoodValid = 0;
				for(int validIdx = 0; validIdx < numValidBatches; validIdx++){
					if(rank == 0){
						for(int i = 1; i < numProcess; i++){
							MPI_Isend(nvValidData->getDevData() + (i-1)*miniDataLen, \
									miniDataLen, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &req);
							MPI_Isend(nvValidLabel->getDevData() + (i-1)*miniLabelLen, \
									miniLabelLen, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &req);
						}
					}
					else{

						MPI_Irecv(nvValidData->getDevData(), miniDataLen, \
								MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &req);
						MPI_Irecv(nvValidLabel->getDevData(), miniLabelLen, \
								MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &req);
					}
					if(rank != 0){
						layer1.computeLogistic(nvValidData, nvValidLabel, false);
						loglihoodValid += layer1.computeError(nvValidLabel, \
								errorValid);
					}
					else{
						nvValidData->changePtr((numProcess - 1) * miniDataLen);
						nvValidLabel->changePtr((numProcess - 1)* miniLabelLen);
					}
				}
				int totalValid;
				MPI_Reduce(&errorValid, &totalValid, 1, MPI_INT, MPI_SUM, \
						0, MPI_COMM_WORLD);
				if(rank == 0){
					cout << "epoch: " << epochIdx 
						<< ",error rate: " << (float)totalValid/validNum  \
						<< ",likelihood: "<< loglihoodValid << endl;
				}
			}
		}
	}
	if(rank == 0){
		t = clock() - t;
		cout << " " << ((float)t/CLOCKS_PER_SEC)/numEpoches << " seconds. \n";
	}
	//				savePars(hHidVis, "../data/pars/hHidVis_t1.bin");
	//  			savePars(hHidBiases, "../data/pars/hHidBiases_t1.bin");
	//				savePars(hAvgout, "../data/pars/hAvgout_t1.bin");
	//				savePars(hOutBiases, "../data/pars/hOutBiases_t1.bin");

	delete nvTrainData;
	delete nvTrainLabel;
	delete nvValidData;
	delete nvValidLabel;

	MPI_Finalize();
	return 0;
}
