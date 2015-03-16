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

enum swapInfo{SWAP_AVGOUT_PUSH, \
	SWAP_AVGOUT_FETCH, \
		SWAP_BIAS_PUSH, \
		SWAP_BIAS_FETCH};

typedef struct _ThreadControlMSG{
	int sendPid;
	int recvPid;
	//传递batchIdx
	int myState;
	enum swapInfo mySwap; 
	float* data;
	bool isMoveDataPos;
	bool isSender;
	int transLen;
} ThreadControlMSG, *pThreadControlMSG;

int rank;
int numProcess;
MPI_Comm newComm[10];

pthread_mutex_t mutexS;
#define THREAD_END 10000

void* watchState(void* msg){
	pThreadControlMSG myMsg = (pThreadControlMSG)msg;
	float* myData = myMsg->data;
	MPI_Request req;
	MPI_Status status;
	int numCompute = -1;
	while(myMsg->myState != THREAD_END){
		if(myMsg->isSender)
			MPI_Recv(&myMsg->myState, 1, MPI_INT, myMsg->recvPid, \
					myMsg->mySwap*100 + (numCompute+1)*(numProcess-1), \
					newComm[myMsg->recvPid], &status);
		else
			MPI_Recv(&myMsg->myState, 1, MPI_INT, myMsg->sendPid, \
					myMsg->mySwap*100 + (numCompute+1)*(numProcess-1), \
					newComm[myMsg->recvPid], &status);
		//			cout <<"send:" << myMsg->mySwap*100 + (numCompute+1) * (numProcess - 1) \
		<< endl;
		if(myMsg->isSender){
			//			pthread_mutex_lock(&mutexS);
			MPI_Send(myData, myMsg->transLen, MPI_FLOAT, myMsg->recvPid, \
					myMsg->mySwap*200 + (numCompute+1) * (numProcess-1), \
					newComm[myMsg->recvPid]);
			//			pthread_mutex_unlock(&mutexS);
		}
		else{
			//			pthread_mutex_lock(&mutexS);
			MPI_Recv(myData, myMsg->transLen, MPI_FLOAT, myMsg->sendPid, \
					myMsg->mySwap*200 + (numCompute+1)*(numProcess-1), \
					newComm[myMsg->sendPid], &status);
			//			pthread_mutex_unlock(&mutexS);
		}
		//		cout <<"send data:" << myMsg->mySwap*200 + (numCompute+1)*(numProcess-1) \
		<< endl;
		numCompute = myMsg->myState;
	}
	pthread_exit(0);
}


//默认创建线程的为发送方
void createAndRun(pthread_t* tid, pThreadControlMSG tMSG, float* data, \
		const int transLen, enum swapInfo mySwap, \
		bool isSender = true){

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
		tMSG[i].data = data;
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

	int proTrainDataLen = logistic->trainNum * inSqrt / (numProcess - 1);
	int proTrainLabelLen = logistic->trainNum / (numProcess - 1);
	int proValidDataLen = logistic->validNum * inSqrt / (numProcess - 1);
	int proValidLabelLen = logistic->validNum / (numProcess - 1);

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

	//直接将input数据scatter给剩下的线程。让它们自己移动，用循环处理
	for(int i = 1; i < numProcess; i++){
		MPI_Send(nvTrainData->getDevData()+(i-1)*proTrainDataLen, proTrainDataLen, \
				MPI_FLOAT, i, i, MPI_COMM_WORLD);
		MPI_Send(nvTrainLabel->getDevData()+(i-1)*proTrainLabelLen, \
				proTrainLabelLen, MPI_FLOAT, i, i, MPI_COMM_WORLD);
		MPI_Send(nvValidData->getDevData()+(i-1)*proValidDataLen, proValidDataLen, \
				MPI_FLOAT, i, i, MPI_COMM_WORLD);
		MPI_Send(nvValidLabel->getDevData()+(i-1)*proValidLabelLen, \
				proValidLabelLen, MPI_FLOAT, i, i, MPI_COMM_WORLD);
	}


	pthread_mutex_init(&mutexS, NULL);
	int openTimes = 4;
	pthread_t openThread[openTimes * (numProcess - 1)];
	pThreadControlMSG tMSG = new ThreadControlMSG[openTimes * (numProcess - 1)];


	//接收更新的参数
	createAndRun(openThread, tMSG, avgOut->getDevData(), \
			avgOutLen, SWAP_AVGOUT_PUSH, false);	
	createAndRun(openThread + (numProcess - 1), tMSG + (numProcess - 1), \
			outBiases->getDevData(), outBiasLen, SWAP_BIAS_PUSH, false);	
	//发送参数
	createAndRun(openThread + (numProcess - 1) * 2, tMSG + (numProcess - 1) * 2, \
			avgOut->getDevData(), avgOutLen, SWAP_AVGOUT_FETCH);	
	createAndRun(openThread + (numProcess - 1) * 3, tMSG + (numProcess - 1) * 3, \
			outBiases->getDevData(), outBiasLen, SWAP_BIAS_FETCH);	

	for(int i = 0; i < openTimes * (numProcess - 1); i++){
		pthread_join(openThread[i], NULL);
	}

	//				savePars(hHidVis, "../data/pars/hHidVis_t1.bin");
	//  			savePars(hHidBiases, "../data/pars/hHidBiases_t1.bin");
	//				savePars(hAvgout, "../data/pars/hAvgout_t1.bin");
	//				savePars(hOutBiases, "../data/pars/hOutBiases_t1.bin");	
	int errorValid = 0;
	int totalValid;
	MPI_Reduce(&errorValid, &totalValid, 1, MPI_INT, MPI_SUM, \
			1, MPI_COMM_WORLD);

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
	int inSqrt = logistic->inSize * logistic->inSize;

	//int hidVisLen = logistic->numFilters * logistic->filterSize \
	* logistic->filterSize;
	//	int hidBiasLen = numFilters;
	int avgOutLen = inSqrt * logistic->numOut;
	int outBiasLen = logistic->numOut;

	int miniDataLen = logistic->minibatchSize * inSqrt;
	int miniLabelLen = logistic->minibatchSize;

	int proTrainDataLen = logistic->trainNum * inSqrt / (numProcess - 1);
	int proTrainLabelLen = logistic->trainNum / (numProcess - 1);
	int proValidDataLen = logistic->validNum * inSqrt / (numProcess - 1);
	int proValidLabelLen = logistic->validNum / (numProcess - 1);

	NVMatrix* avgOut;
	NVMatrix* outBiases;

	NVMatrix* nvTrainData = new NVMatrix(logistic->trainNum / (numProcess - 1), \
			inSqrt, NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	NVMatrix* nvValidData = new NVMatrix(logistic->validNum / (numProcess - 1), \
			inSqrt, NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	NVMatrix* nvTrainLabel = new NVMatrix(logistic->trainNum / (numProcess - 1), 1, \
			NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	NVMatrix* nvValidLabel = new NVMatrix(logistic->validNum / (numProcess - 1), 1, \
			NVMatrix::ALLOC_ON_UNIFIED_MEMORY);

	NVMatrix* miniData = new NVMatrix(nvTrainData->getDevData(), \
			logistic->minibatchSize, inSqrt);
	NVMatrix* miniLabel = new NVMatrix(nvTrainLabel->getDevData(), \
			logistic->minibatchSize, 1);

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

	MPI_Request reqs;
	MPI_Status status;

	MPI_Recv(nvTrainData->getDevData(), proTrainDataLen, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);
	MPI_Recv(nvTrainLabel->getDevData(), proTrainLabelLen, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);
	MPI_Recv(nvValidData->getDevData(), proValidDataLen, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);
	MPI_Recv(nvValidLabel->getDevData(), proValidLabelLen, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);


	int passMsg;	
	char* myBuffer = new char[logistic->numOut * ( 1+inSqrt ) * sizeof(float) * 50];
	int detachOut;
	int detachS;
	MPI_Buffer_attach(myBuffer, logistic->numOut * ( 1+inSqrt) * sizeof(float) * 50);
	cout << "create thread\n";

	clock_t t;
	t = clock();
	for(int epochIdx = 0; epochIdx < logistic->numEpoches; epochIdx++){
		int error = 0;	

		for(int batchIdx = 0; batchIdx < logistic->numMinibatches/(numProcess-1); \
				batchIdx++){
			if(batchIdx == logistic->numMinibatches - 1){ 
				if(epochIdx == logistic->numEpoches - 1)
					passMsg = THREAD_END;    
				else
					passMsg = -1;
			}
			else
				passMsg = batchIdx;
cout << rank<< ":done1\n";
			nvTrainData->slice(miniData->getDevData(), batchIdx * miniDataLen);
			nvTrainLabel->slice(miniLabel->getDevData(), batchIdx * miniLabelLen);

cout << rank<< ":done2\n";
			layer1.computeLogistic(miniData, miniLabel, true);
			layer1.computeError(miniLabel, error);

			avgOut = layer1.getAvgOut();
			outBiases = layer1.getOutBias();

cout << rank<< ":done3\n";
			if((batchIdx + 1) % logistic->nPush == 0){
				MPI_Send(&passMsg, 1, MPI_INT, 0, \
						SWAP_AVGOUT_PUSH*100 + batchIdx*(numProcess-1), \
						newComm[rank]);

				MPI_Send((rank-1)*avgOutLen/(numProcess-1) \
						+ avgOut->getDevData(), avgOutLen/(numProcess-1), \
						MPI_FLOAT, 0, SWAP_AVGOUT_PUSH * 200 \
						+ batchIdx * (numProcess - 1), \
						newComm[rank]);

				MPI_Send(&passMsg, 1, MPI_INT, 0, \
						SWAP_BIAS_PUSH*100 + batchIdx*(numProcess-1), \
						newComm[rank]);
				MPI_Send((rank-1)*outBiasLen/(numProcess-1) \
						+ outBiases->getDevData(), outBiasLen/(numProcess-1), \
						MPI_FLOAT, 0, SWAP_BIAS_PUSH*200 + batchIdx*(numProcess-1), \
						newComm[rank]);
			}

cout << rank<< ":done4\n";
			if((batchIdx + 1) % logistic->nFetch == 0){
				MPI_Send(&passMsg, 1, MPI_INT, 0, \
						SWAP_AVGOUT_FETCH*100+batchIdx*(numProcess-1), \
						newComm[rank]);
				MPI_Recv(avgOut->getDevData(), avgOutLen, MPI_FLOAT, 0, \
						SWAP_AVGOUT_FETCH*200+batchIdx*(numProcess-1), \
						newComm[rank], &status);

				MPI_Send(&passMsg, 1, MPI_INT, 0, \
						SWAP_BIAS_FETCH*100+batchIdx*(numProcess-1), \
						newComm[rank]);
				MPI_Recv(outBiases->getDevData(), outBiasLen, MPI_FLOAT, \
						0, SWAP_BIAS_FETCH*200+batchIdx*(numProcess-1), \
						newComm[rank], &status);
			}

cout << rank<< ":done5\n";
			if(batchIdx == logistic->numMinibatches - 1){
				int errorValid = 0;
				float loglihoodValid = 0;
				for(int validIdx = 0; validIdx < logistic->numValidBatches / (numProcess - 1); validIdx++){

					nvValidData->slice(miniData->getDevData(), validIdx * miniDataLen);
					nvValidLabel->slice(miniLabel->getDevData(), validIdx * miniLabelLen);

					layer1.computeLogistic(miniData, miniLabel, false);
					loglihoodValid += layer1.computeError(miniLabel, \
							errorValid);
				}
				int totalValid;
				MPI_Reduce(&errorValid, &totalValid, 1, MPI_INT, MPI_SUM, \
						1, MPI_COMM_WORLD);
				if(rank == 1){
					cout << "epochIdx: " << epochIdx << ",error: " \
						<< ((float)errorValid*(numProcess-1))/logistic->validNum << endl;
					//					<< ",likelihood: "<< loglihood<< endl;
				}
			}
		}

	}
	detachS = MPI_Buffer_detach(myBuffer, &detachOut);
	cout << (detachS == 0 ? "success\n" : "failed\n" )<< endl;

	if(rank == 1){
		t = clock() - t;
		cout << " " << ((float)t/CLOCKS_PER_SEC)/logistic->numEpoches << " seconds. \n";
	}

	delete nvTrainData;
	delete nvTrainLabel;
	delete nvValidData;
	delete nvValidLabel;
}

int main(int argc, char** argv){

	int prov;
	MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE, &prov);
	if (prov < MPI_THREAD_MULTIPLE)
	{
		printf("Error: the MPI library doesn't provide the required thread level\n");
		MPI_Abort(MPI_COMM_WORLD, 0);
	}
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
	logistic->numEpoches = 20; 
	logistic->nPush = 1;
	logistic->nFetch = 1;
	
cout << "start\n";
	if(rank == 0){
		for(int i = 1; i < numProcess; i++)
			MPI_Comm_split(MPI_COMM_WORLD, i, rank, &newComm[i]);
		cout << "split done1\n";
		managerNode(logistic);
	}
	else{
		MPI_Comm_split(MPI_COMM_WORLD, rank, rank, &newComm[rank]);
		cout << "split done1\n";
		workerNode(logistic);
	}

	delete logistic;
	for(int i = 1; i < numProcess; i++)
		MPI_Comm_free(&newComm[i]);
	MPI_Finalize();
	return 0;
}
