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

enum swapInfo{SWAP_PIXEL_TRAIN, SWAP_PIXEL_TRAIN_REQUEST,\
	SWAP_PIXEL_VALID, SWAP_PIXEL_VALID_REQUEST, \
		SWAP_LABEL_TRAIN, SWAP_LABEL_TRAIN_REQUEST, \
		SWAP_LABEL_VALID, SWAP_LABEL_VALID_REQUEST, \
		SWAP_AVGOUT_PUSH, SWAP_AVGOUT_PUSH_REQUEST, \
		SWAP_AVGOUT_FETCH, SWAP_AVGOUT_FETCH_REQUEST, \
		SWAP_BIAS_PUSH, SWAP_BIAS_PUSH_REQUEST, \
		SWAP_BIAS_FETCH, SWAP_BIAS_FETCH_REQUEST};

typedef struct _ThreadControlMSG{
	int sendPid;
	int recvPid;
	//传递batchIdx
	int myState;
	enum swapInfo mySwap; 
	float* data;
	bool isMoveDataPos;
	bool isSender;
	int moveLen;
	int moveTime;
	int transLen;
	_ThreadControlMSG():isMoveDataPos(false) {}
} ThreadControlMSG, *pThreadControlMSG;

int rank;
int numProcess;
pthread_mutex_t mutexS;

#define THREAD_END 10000

void* watchState(void* msg){
	pThreadControlMSG myMsg = (pThreadControlMSG)msg;
	float* myData = myMsg->data;
	MPI_Request req;
	MPI_Status status;
	int numCompute = 0;
	while(myMsg->myState != THREAD_END){
		pthread_mutex_lock(&mutexS);
		if(myMsg->isSender)
			MPI_Recv(&myMsg->myState, 1, MPI_INT, myMsg->recvPid, \
					myMsg->mySwap + 1, MPI_COMM_WORLD, &status);
		else
			MPI_Recv(&myMsg->myState, 1, MPI_INT, myMsg->sendPid, \
					myMsg->mySwap + 1, MPI_COMM_WORLD, &status);
	cout <<rank << ":recv:" << myMsg->recvPid << ":" << myMsg->myState \
				<< ":"<< myMsg->transLen<< endl;
		if(myMsg->isSender){
			MPI_Send(myData, myMsg->transLen, MPI_FLOAT, myMsg->recvPid, \
					myMsg->mySwap + myMsg->myState * (numProcess - 1), \
					MPI_COMM_WORLD);
		}
		else
			MPI_Recv(myData, myMsg->transLen, MPI_FLOAT, myMsg->sendPid, \
					myMsg->mySwap + myMsg->myState * (numProcess - 1), \
					MPI_COMM_WORLD, &status);
		pthread_mutex_unlock(&mutexS);
cout << rank << ":send:" << myMsg->recvPid <<":"<< myMsg->myState \
			<< ":"<< myMsg->transLen<< endl;
		if(myMsg->isMoveDataPos){
			if(myMsg->myState < (myMsg->moveTime - 1)){
				numCompute++;
				myData += myMsg->moveLen;
			}else{
				//input数据回到起始位置
				numCompute = 0;
				myData = myMsg->data;
			}
		}
	}
	pthread_exit(0);
}


//默认创建线程的为发送方
void createAndRun(pthread_t* tid, pThreadControlMSG tMSG, const int numProcess, \
		float* data, const int transLen, enum swapInfo mySwap, \
		bool isSender = true, bool isMoveDataPos = false, const int moveTime = 0){

	for(int i = 0; i < numProcess - 1; i++){
		if(isSender){
			tMSG[i].sendPid = 0;
			tMSG[i].recvPid = i + 1;
			tMSG[i].isSender = true;
		}else{
			tMSG[i].sendPid = i + 1;
			tMSG[i].recvPid = 0;
			tMSG[i].isSender = false;
		}
		if(isMoveDataPos){
			tMSG[i].data = data + transLen * i;
			tMSG[i].isMoveDataPos = true;
			tMSG[i].moveLen = (numProcess - 1) * transLen;
			tMSG[i].moveTime = moveTime;
		}else{
			tMSG[i].data = data;
		}
		tMSG[i].mySwap = mySwap;
		tMSG[i].transLen = transLen;
		int error = pthread_create(&tid[i], NULL, \
				watchState, (void*)&tMSG[i]);
		if(error){
			cout << "Error - pthread_create() return code: " << error << endl;
			exit(EXIT_FAILURE);
		}
	}
}

void managerNode(pars* logistic){

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

	clock_t t;
	t = clock();


	pthread_mutex_init(&mutexS, NULL);
	int openTimes = 1;
	pthread_t openThread[openTimes * (numProcess - 1)];
	pthread_t openThread1[openTimes * (numProcess - 1)];
	pThreadControlMSG tMSG = new ThreadControlMSG[openTimes * (numProcess - 1)];
	pThreadControlMSG tMSG1 = new ThreadControlMSG[openTimes * (numProcess - 1)];

	createAndRun(openThread1, tMSG1, numProcess, nvTrainData->getDevData(), \
			miniDataLen, SWAP_PIXEL_TRAIN, \
			true, true, logistic->numMinibatches);	
	createAndRun(openThread, tMSG, numProcess, nvTrainLabel->getDevData(), \
			miniLabelLen, SWAP_LABEL_TRAIN, \
			true, true, logistic->numMinibatches);
	//接收更新的参数
//	createAndRun(openThread + 4, tMSG[2], numProcess, avgOut->getDevData(), \
			avgOutLen, SWAP_AVGOUT_PUSH, false);	
//	createAndRun(openThread + 6, tMSG[3], numProcess, outBiases->getDevData(), \
			outBiasLen, SWAP_BIAS_PUSH, false);	
	/*	//发送参数
		createAndRun(openThread[4], tMSG[4], numProcess, avgOut->getDevData(), \
		avgOutLen, SWAP_AVGOUT_FETCH);	
		createAndRun(openThread[5], tMSG[5], numProcess, outBiases->getDevData(), \
		outBiasLen, SWAP_BIAS_FETCH);	
	 */
	for(int i = 0; i < openTimes * (numProcess - 1); i++){
		pthread_join(openThread[i], NULL);
		pthread_join(openThread1[i], NULL);
	}

	t = clock() - t;
	cout << " " << ((float)t/CLOCKS_PER_SEC)/logistic->numEpoches << " seconds. \n";
	//				savePars(hHidVis, "../data/pars/hHidVis_t1.bin");
	//  			savePars(hHidBiases, "../data/pars/hHidBiases_t1.bin");
	//				savePars(hAvgout, "../data/pars/hAvgout_t1.bin");
	//				savePars(hOutBiases, "../data/pars/hOutBiases_t1.bin");	

	delete[] tMSG;


	delete nvTrainData;
	delete nvTrainLabel;
	delete nvValidData;
	delete nvValidLabel;
	delete avgOut;
	delete outBiases;

	delete hHidVis;
	delete hHidBiases;
	delete hAvgout;
	delete hOutBiases;

	pthread_mutex_destroy(&mutexS);
}

void workerNode(pars* logistic){
	/*
	   int rank;
	   int numProcess;
	   MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	   MPI_Comm_size(MPI_COMM_WORLD,&numProcess);
	 */
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

	MPI_Request reqs[2];
	MPI_Status status[2];

	int passMsg[2];	

	for(int epochIdx = 0; epochIdx < logistic->numEpoches; epochIdx++){
		int error = 0;	
		for(int batchIdx = 0; batchIdx < logistic->numMinibatches; batchIdx++){
			passMsg[0] = batchIdx;
			passMsg[1] = batchIdx;
			if(epochIdx == logistic->numEpoches - 1 && \
					batchIdx == logistic->numMinibatches - 1){
				passMsg[0] = THREAD_END;			
				passMsg[1] = THREAD_END;			
			}

			MPI_Send(&passMsg[1], 1, MPI_INT, 0, SWAP_LABEL_TRAIN_REQUEST, \
					MPI_COMM_WORLD);
			MPI_Recv(nvTrainLabel->getDevData(), miniLabelLen, MPI_FLOAT, 0, \
					SWAP_LABEL_TRAIN + passMsg[1] * (numProcess - 1), \
					MPI_COMM_WORLD, &status[1]);		
//			if(rank == 1)
				cout << rank  << ":receive label:" \
					<< ":" << passMsg[0]  \
					<<":" << miniLabelLen<< endl;

			MPI_Send(&passMsg[0], 1, MPI_INT, 0, SWAP_PIXEL_TRAIN_REQUEST, \
					MPI_COMM_WORLD);
			MPI_Recv(nvTrainData->getDevData(), miniDataLen, MPI_FLOAT, 0, \
					SWAP_PIXEL_TRAIN + passMsg[0] * (numProcess - 1), \
					MPI_COMM_WORLD, &status[0]);
//			if(rank == 1)
				cout << rank  << ":receive pixel:" \
					<< ":" << passMsg[0]  \
					<<":" << miniDataLen<< endl;
			//			MPI_Irecv(nvTrainData->getDevData(), miniDataLen, MPI_FLOAT, 0, \
			SWAP_PIXEL_TRAIN, MPI_COMM_WORLD, &reqs[0]);
			//			MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
			//if(rank == 2)
			//			MPI_Irecv(nvTrainLabel->getDevData(), miniLabelLen, MPI_FLOAT, 0, \
			SWAP_LABEL_TRAIN, MPI_COMM_WORLD, &reqs[1]);
			//			MPI_Wait(&reqs[1], MPI_STATUS_IGNORE);



			layer1.computeLogistic(nvTrainData, nvTrainLabel, true);
			layer1.computeError(nvTrainLabel, error);

			avgOut = layer1.getAvgOut();
			outBiases = layer1.getOutBias();
/*			if((batchIdx + 1) % logistic->nPush == 0){
				MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_AVGOUT_PUSH_REQUEST, \
						MPI_COMM_WORLD);
				MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_BIAS_PUSH_REQUEST, \
						MPI_COMM_WORLD);
				MPI_Send((rank-1)*avgOutLen/(numProcess-1) \
						+ avgOut->getDevData(), avgOutLen/(numProcess-1), \
						MPI_FLOAT, 0, SWAP_AVGOUT_PUSH + 2 * batchIdx, MPI_COMM_WORLD);
				MPI_Send((rank-1)*outBiasLen/(numProcess-1) \
						+ outBiases->getDevData(), outBiasLen/(numProcess-1), \
						MPI_FLOAT, 0, SWAP_BIAS_PUSH, MPI_COMM_WORLD);
				//				MPI_Isend((rank-1)*avgOutLen/(numProcess-1) \
				+ avgOut->getDevData(), avgOutLen/(numProcess-1), \
					MPI_FLOAT, 0, SWAP_AVGOUT_PUSH, MPI_COMM_WORLD, &reqs[0]);
				//				MPI_Isend((rank-1)*outBiasLen/(numProcess-1) \
				+ outBiases->getDevData(), outBiasLen/(numProcess-1), \
					MPI_FLOAT, 0, SWAP_BIAS_PUSH, MPI_COMM_WORLD, &reqs[1]);
			}
*/			/*
			   if((batchIdx + 1) % logistic->nFetch == 0){
			   MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_AVGOUT_FETCH, \
			   MPI_COMM_WORLD);
			   MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_BIAS_FETCH, \
			   MPI_COMM_WORLD);
			   MPI_Recv(avgOut->getDevData(), avgOutLen, MPI_FLOAT, 0, \
			   SWAP_AVGOUT_FETCH, MPI_COMM_WORLD, &status);
			   MPI_Recv(outBiases->getDevData(), outBiasLen, MPI_FLOAT, \
			   0, SWAP_BIAS_FETCH, MPI_COMM_WORLD, &status);
			//	MPI_Irecv(avgOut->getDevData(), avgOutLen, MPI_FLOAT, 0, \
			SWAP_AVGOUT_FETCH, MPI_COMM_WORLD, &reqs[0]);
			//	MPI_Irecv(outBiases->getDevData(), outBiasLen, MPI_FLOAT, \
			0, SWAP_BIAS_FETCH, MPI_COMM_WORLD, &reqs[1]);
			//	MPI_Waitall(2, reqs, status);
			}*/
		}
		//			if(rank == 1){
		//			cout << "epochIdx: " << epochIdx << ",error: " \
		//			<< (float)error*2/logistic->trainNum << endl;
		//			<< ",likelihood: "<< loglihood<< endl;
		//		}
	}


	/*		int errorValid = 0;
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
	 */
	delete nvTrainData;
	delete nvTrainLabel;
	delete nvValidData;
	delete nvValidLabel;
}

int main(int argc, char** argv){

	//	int rank;
	//	int numProcess;

	int prov;
	MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE, &prov);
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
	logistic->numEpoches = 10; 
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
