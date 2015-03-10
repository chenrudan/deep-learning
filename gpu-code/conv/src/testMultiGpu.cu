/*
 * filename:testMultiGpu.cu
 */

#include <iostream>
#include <fstream>
#include <time.h>
#include <cmath>
#include <pthread.h>
#include "mpi.h"
#include "utils.h"
#include "matrix.h"
#include "nvmatrix.cuh"
#include "convnet.cuh"
#include "convnet_kernel.cuh"

using namespace std;

enum state{WORK, STOP, END};

typedef struct _ThreadControlMSG{
	int sendPid;
	int recvPid;
	enum state myState;
	float* data;
	bool isMoveDataPos;
	int moveLen;
	int moveiTime;
	int transLen;
	threadControlMSG(): isMoveDataPos(false);
} ThreadControlMSG, *pThreadControlMSG;

void* watchSend(void* msg){
	pThreadControlMSG myMsg = (pthreadControlMSG)msg;
	float* myData = myMsg->data;
	MPI_Request req;
	int numCompute = 0;
	while(true){
		MPI_Irecv(&myMsg->myState, 1, MPI_INT, sendPid, 0, MPI_COMM_WORLD, &req);
		if(myMsg->myData != END)
			MPI_Isend(&myData, transLen, MPI_FLOAT, sendPid, 0, \
						MPI_COMM_WORLD, &req);
			if(myMsg->isMoveDataPos){
				if()
			}
	}
}


void managerNode(pars* logistic){

	int rank;
	int numProcess;
	MPI_Request req;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&numProcess);

	float* trainDataPtr;
	float* trainLabelPtr;
	float* validDataPtr;
	float* validLabelPtr;

	int inSqrt = logistic->inSize * logistic->inSize;
	int hidVisLen = logistic->numFilters * logistic->filterSize \
					* logistic->filterSize;
	//	int hidBiasLen = numFilters;
	int avgOutLen = inSqrt * logistic->numOut;
	int outBiasLen = logistic->numOut;
	int miniDataLen = logistic->minibatchSize * inSqrt;
	int miniLabelLen = logistic->minibatchSize;

	NVMatrix* nvTrainData = new NVMatrix(logistic->trainNum, inSqrt, 
			NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	NVMatrix* nvValidData = new NVMatrix(logistic->validNum, inSqrt, \
			NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	NVMatrix* nvTrainLabel = new NVMatrix(logistic->trainNum, 1, \
			NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	NVMatrix* nvValidLabel = new NVMatrix(logistic->validNum, 1, \
			NVMatrix::ALLOC_ON_UNIFIED_MEMORY);

	NVMatrix* avgOut = new NVMatrix(inSqrt, logistic->numOut, \
			NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	NVMatrix* outBiases = new NVMatrix(1, logistic->numOut, \
			NVMatrix::ALLOC_ON_UNIFIED_MEMORY);

	cout << "=========================\n" \
		<< "train: " << logistic->trainNum \
		<< "\nvalid: " << logistic->validNum \
		<< "\nfiltersize: " << logistic->filterSize \
		<< "\nnumFilters: " << logistic->numFilters \
		<< "\nepsHidVis: " << logistic->epsHidVis \
		<< "\nepsHidBias: " << logistic->epsHidBias \
		<< "\nepsAvgOut: " << logistic->epsAvgOut \
		<< "\nepsOutBias: " << logistic->epsOutBias \
		<< "\nmom: " << logistic->mom \
		<< "\nwcHidVis: " << logistic->wcHidVis \
		<< "\nwcAvgOut: " << logistic->wcAvgOut \
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

	Matrix* hHidVis = new Matrix(logistic->numFilters, logistic->filterSize \
			* logistic->filterSize);
	Matrix* hHidBiases = new Matrix(logistic->numFilters, 1);
	Matrix* hAvgout = new Matrix(inSqrt, logistic->numOut);
	Matrix* hOutBiases = new Matrix(1, logistic->numOut);

	//0号进程初始化参数，进行分发
	initW(hHidVis->getData(), hidVisLen);
	memset(hHidBiases->getData(), 0, sizeof(float) * logistic->numFilters);
	memset(hAvgout->getData(), 0, sizeof(float) * avgOutLen);
	memset(hOutBiases->getData(), 0, sizeof(float) * logistic->numOut);
	//	readPars(hHidVis, "hHidVis_t1.bin");
	//	readPars(hHidBiases, "hHidBiases_t1.bin");
	//	readPars(hAvgout, "hAvgout_t1.bin");
	//	readPars(hOutBiases, "hOutBiases_t1.bin");

	MPI_Bcast(hAvgout->getData(), avgOutLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(hOutBiases->getData(), outBiasLen, MPI_FLOAT, 0, MPI_COMM_WORLD);

	pthread_t tid;
	int error = pthread_create(&tid, NULL, )

	clock_t t;
	t = clock();
	for(int epochIdx = 0; epochIdx < logistic->numEpoches; epochIdx++){

		nvTrainData->setPtr(trainDataPtr);
		nvTrainLabel->setPtr(trainLabelPtr);
		nvValidData->setPtr(validDataPtr);
		nvValidLabel->setPtr(validLabelPtr);
		for(int batchIdx = 0; batchIdx < logistic->numMinibatches; batchIdx++){
			//发送输入

			for(int i = 1; i < numProcess; i++){
				MPI_Isend(nvTrainData->getDevData() + (i-1)*miniDataLen, \
						miniDataLen, MPI_FLOAT, i, i, MPI_COMM_WORLD, &req);
				MPI_Isend(nvTrainLabel->getDevData() + (i-1)*miniLabelLen, \
						miniLabelLen, MPI_FLOAT, i, i, MPI_COMM_WORLD, &req);
				//		cout <<  "rank: " << i<< ":req"<< req << "\n";
			}

			//			cout << "done1\n";
			nvTrainData->changePtr((numProcess-1) * miniDataLen);
			nvTrainLabel->changePtr((numProcess-1) * miniLabelLen);

			//接收部分更新的参数
			if((batchIdx + 1) % logistic->nPush == 0){
				for(int sender = 1; sender < numProcess; sender++){
					MPI_Irecv((sender-1)*avgOutLen/(numProcess-1) \
							+ avgOut->getDevData(), avgOutLen/(numProcess-1), \
							MPI_FLOAT, sender, sender, MPI_COMM_WORLD, \
							&req);
					MPI_Wait(&req,&status);
					MPI_Irecv((sender-1)*outBiasLen/(numProcess-1) \
							+ outBiases->getDevData(), outBiasLen/(numProcess-1), \
							MPI_FLOAT, sender, sender, MPI_COMM_WORLD, \
							&req);
					MPI_Wait(&req,&status);
				}
			}
			//发送所有参数
			if((batchIdx + 1) % logistic->nFetch == 0){
				for(int i = 1; i < numProcess; i++){
					MPI_Isend(avgOut->getDevData(), avgOutLen, MPI_FLOAT, \
							i, i, MPI_COMM_WORLD, &req);
					MPI_Isend(outBiases->getDevData(), outBiasLen, MPI_FLOAT, \
							i, i, MPI_COMM_WORLD, &req);
				}
			}
		}
		int errorValid = 0;
		float loglihoodValid = 0;
		for(int validIdx = 0; validIdx < logistic->numValidBatches; validIdx++){
			for(int i = 1; i < numProcess; i++){
				MPI_Isend(nvValidData->getDevData() + (i-1)*miniDataLen, \
						miniDataLen, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &req);
				MPI_Isend(nvValidLabel->getDevData() + (i-1)*miniLabelLen, \
						miniLabelLen, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &req);
			}
			nvValidData->changePtr((numProcess - 1) * miniDataLen);
			nvValidLabel->changePtr((numProcess - 1)* miniLabelLen);

		}	
		int totalValid;
		MPI_Reduce(&errorValid, &totalValid, 1, MPI_INT, MPI_SUM, \
				0, MPI_COMM_WORLD);
		cout << "epoch: " << epochIdx 
			<< ",error rate: " << (float)totalValid/logistic->validNum  \
			<< ",likelihood: "<< loglihoodValid << endl;
	}
	t = clock() - t;
	cout << " " << ((float)t/CLOCKS_PER_SEC)/logistic->numEpoches << " seconds. \n";
	//				savePars(hHidVis, "../data/pars/hHidVis_t1.bin");
	//  			savePars(hHidBiases, "../data/pars/hHidBiases_t1.bin");
	//				savePars(hAvgout, "../data/pars/hAvgout_t1.bin");
	//				savePars(hOutBiases, "../data/pars/hOutBiases_t1.bin");
	delete nvTrainData;
	delete nvTrainLabel;
	delete nvValidData;
	delete nvValidLabel;
	delete avgOut;
	delete outBiases;
}

void workerNode(pars* logistic){

	int rank;
	int numProcess;
	MPI_Request req;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&numProcess);

	int inSqrt = logistic->inSize * logistic->inSize;

	//int hidVisLen = logistic->numFilters * logistic->filterSize \
	* logistic->filterSize;
	//	int hidBiasLen = numFilters;
	int avgOutLen = inSqrt * logistic->numOut;
	int outBiasLen = logistic->numOut;

	int miniDataLen = logistic->minibatchSize * inSqrt;
	int miniLabelLen = logistic->minibatchSize;

	NVMatrix* avgOut;
	NVMatrix* outBiases;

	NVMatrix* nvTrainData = new NVMatrix(logistic->minibatchSize, inSqrt, \
			NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	NVMatrix* nvValidData = new NVMatrix(logistic->minibatchSize, inSqrt, \
			NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	NVMatrix* nvTrainLabel = new NVMatrix(logistic->minibatchSize, 1, \
			NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	NVMatrix* nvValidLabel = new NVMatrix(logistic->minibatchSize, 1, \
			NVMatrix::ALLOC_ON_UNIFIED_MEMORY);

	Matrix* hHidVis = new Matrix(logistic->numFilters, logistic->filterSize \
			* logistic->filterSize);
	Matrix* hHidBiases = new Matrix(logistic->numFilters, 1);
	Matrix* hAvgout = new Matrix(inSqrt, logistic->numOut);
	Matrix* hOutBiases = new Matrix(1, logistic->numOut);

	MPI_Bcast(hAvgout->getData(), avgOutLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(hOutBiases->getData(), outBiasLen, MPI_FLOAT, 0, MPI_COMM_WORLD);

	ConvNet layer1(hHidVis, hAvgout, hHidBiases, hOutBiases, logistic);
	layer1.initCuda();

	//	double loglihood = 0;
	for(int epochIdx = 0; epochIdx < logistic->numEpoches; epochIdx++){
		int error = 0;	
		for(int batchIdx = 0; batchIdx < logistic->numMinibatches; batchIdx++){
			MPI_Irecv(nvTrainData->getDevData(), miniDataLen, MPI_FLOAT, 0, \
					rank, MPI_COMM_WORLD, &req);
			MPI_Wait(&req,&status);
			MPI_Irecv(nvTrainLabel->getDevData(), miniLabelLen, MPI_FLOAT, 0, \
					rank, MPI_COMM_WORLD, &req);
			MPI_Wait(&req,&status);

			layer1.computeLogistic(nvTrainData, nvTrainLabel, true);
			layer1.computeError(nvTrainLabel, error);

			avgOut = layer1.getAvgOut();
			outBiases = layer1.getOutBias();
			if((batchIdx + 1) % logistic->nPush == 0){
				MPI_Isend((rank-1)*avgOutLen/(numProcess-1) \
						+ avgOut->getDevData(), avgOutLen/(numProcess-1), \
						MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &req);
				MPI_Isend((rank-1)*outBiasLen/(numProcess-1) \
						+ outBiases->getDevData(), outBiasLen/(numProcess-1), \
						MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &req);
			}
			if((batchIdx + 1) % logistic->nFetch == 0){
				MPI_Irecv(avgOut->getDevData(), avgOutLen, MPI_FLOAT, 0, rank, \
						MPI_COMM_WORLD, &req);
				MPI_Wait(&req,&status);
				MPI_Irecv(outBiases->getDevData(), outBiasLen, MPI_FLOAT, \
						0, rank, MPI_COMM_WORLD, &req);
				MPI_Wait(&req,&status);
			}
		}
		/*	if(rank == 1){
			cout << "epochIdx: " << epochIdx << ",error: " \
			<< (float)error*2/logistic->trainNum \
			<< ",likelihood: "<< loglihood<< endl;*/
		int errorValid = 0;
		float loglihoodValid = 0;
		for(int validIdx = 0; validIdx < logistic->numValidBatches; validIdx++){
			MPI_Irecv(nvValidData->getDevData(), miniDataLen, \
					MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &req);
			MPI_Wait(&req,&status);
			MPI_Irecv(nvValidLabel->getDevData(), miniLabelLen, \
					MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &req);
			MPI_Wait(&req,&status);

			layer1.computeLogistic(nvValidData, nvValidLabel, false);
			loglihoodValid += layer1.computeError(nvValidLabel, \
					errorValid);

		}
		int totalValid;
		MPI_Reduce(&errorValid, &totalValid, 1, MPI_INT, MPI_SUM, \
				0, MPI_COMM_WORLD);
	}
	delete nvTrainData;
	delete nvTrainLabel;
	delete nvValidData;
	delete nvValidLabel;
}

int main(int argc, char** argv){

	int rank;
	int numProcess;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&numProcess);

	//检测有几个gpu
	int numGpus;
	cudaGetDeviceCount(&numGpus);
	cudaSetDevice(rank%numGpus);

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
	logistic->numMinibatches = logistic->trainNum / (logistic->minibatchSize \
			* (numProcess - 1));
	logistic->numValidBatches = logistic->validNum / (logistic->minibatchSize \
			* (numProcess - 1));
	logistic->numEpoches = 100; 
	logistic->nPush = 1;
	logistic->nFetch = 1;


	if(rank == 0)
		managerNode(logistic);
	else
		workerNode(logistic);

	delete logistic;
	MPI_Finalize();
	return 0;
}
