/*
 * filename:main.cu
 */

#include <iostream>
#include <fstream>
#include <time.h>
#include <cmath>
#include <omp.h>
#include "mpi.h"
#include "matrix.h"
#include "nvmatrix.cuh"
#include "convnet.cuh"
#include "convnet_kernel.cuh"
#include "utils.h"
#include "logistic.cuh"

using namespace std;

#define THREAD_END 100000
enum swapInfo{SWAP_HIDVIS_PUSH, SWAP_HIDBIAS_PUSH, \
	SWAP_AVGOUT_PUSH, SWAP_OUTBIAS_PUSH,
	SWAP_HIDVIS_FETCH, SWAP_HIDBIAS_FETCH, \
		SWAP_AVGOUT_FETCH, SWAP_OUTBIAS_FETCH};

int numProcess;
int rank;

void managerNode(pars* cnn, pars* logistic){

	cout << "=========================\n" \
		<< "train: " << cnn->trainNum \
		<< "\nvalid: " << cnn->validNum \
		<< "\nfiltersize: " << cnn->filterSize \
		<< "\nnumFilters: " << cnn->numFilters \
		<< "\nepsHidVis: " << cnn->epsHidVis \
		<< "\nepsHidBias: " << cnn->epsHidBias \
		<< "\nepsAvgOut: " << logistic->epsAvgOut \
		<< "\nepsOutBias: " << logistic->epsOutBias \
		<< "\nmom: " << cnn->mom \
		<< "\nwcHidVis: " << cnn->wcHidVis \
		<< "\nwcAvgOut: " << logistic->wcAvgOut << endl;

	int inLen = cnn->inSize * cnn->inSize * cnn->inChannel;
	int hidVisLen = cnn->numFilters * cnn->filterSize \
			* cnn->filterSize;
	int hidBiasLen = cnn->numFilters * 1;
	int avgOutLen = cnn->poolResultSize * cnn->poolResultSize * cnn->numFilters * logistic->numOut;
	int outBiasLen = logistic->numOut;

	int proTrainDataLen = cnn->trainNum * inLen / (numProcess - 1);
	int proTrainLabelLen = cnn->trainNum / (numProcess - 1);
	int proValidDataLen = cnn->validNum * inLen / (numProcess - 1);
	int proValidLabelLen = cnn->validNum / (numProcess - 1);

	NVMatrix* nvTrainData = new NVMatrix(cnn->trainNum, inLen);
//					NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	NVMatrix* nvValidData = new NVMatrix(cnn->validNum, inLen);
//					NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	NVMatrix* nvTrainLabel = new NVMatrix(cnn->trainNum, 1);
//					NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	NVMatrix* nvValidLabel = new NVMatrix(cnn->validNum, 1);
//					NVMatrix::ALLOC_ON_UNIFIED_MEMORY);

	readData(nvTrainData, "../data/input/mnist_train.bin", true);
	readData(nvValidData, "../data/input/mnist_valid.bin", true);
	readData(nvTrainLabel, "../data/input/mnist_label_train.bin", false);
	readData(nvValidLabel, "../data/input/mnist_label_valid.bin", false);

	Matrix* hHidVis = new Matrix(cnn->numFilters, cnn->filterSize * cnn->filterSize);
	Matrix* hHidBiases = new Matrix(cnn->numFilters, 1);
	Matrix* hAvgout = new Matrix(avgOutLen / logistic->numOut, logistic->numOut);
	Matrix* hOutBiases = new Matrix(1, logistic->numOut);

	NVMatrix* hidVis = new NVMatrix(cnn->numFilters, \
			cnn->filterSize * cnn->filterSize);
	NVMatrix* hidBiases = new NVMatrix(cnn->numFilters, 1);
	NVMatrix* avgOut = new NVMatrix(avgOutLen / logistic->numOut, logistic->numOut);
	NVMatrix* outBiases = new NVMatrix(1, logistic->numOut);

	initW(hHidVis->getData(), cnn->numFilters * cnn->filterSize * cnn->filterSize);
	memset(hHidBiases->getData(), 0, sizeof(float) * cnn->numFilters);
	memset(hAvgout->getData(), 0, sizeof(float) * avgOutLen);
	memset(hOutBiases->getData(), 0, sizeof(float) * logistic->numOut);

	//	readPars(hHidVis, "hHidVis_t1.bin");
	//	readPars(hHidBiases, "hHidBiases_t1.bin");
	//	readPars(hAvgout, "hAvgout_t1.bin");
	//	readPars(hOutBiases, "hOutBiases_t1.bin");

	MPI_Bcast(hHidVis->getData(), hidVisLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(hHidBiases->getData(), hidBiasLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(hAvgout->getData(), avgOutLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(hOutBiases->getData(), outBiasLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
/*
float* send = new float[proTrainDataLen];
for(int i =0; i < proTrainDataLen; i++){
	send[i] = 1;
}*/
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

	//pro进程，每个进程进行的数据交换次数，0123是push，4567是fetch
	//4个数据地址，8个线程来分别实现两种操作
	const int transOPTimesInPro = 8;
	const int numDataType = 4;
	float* myData[numDataType] = {hidVis->getDevData(), hidBiases->getDevData(), \
		avgOut->getDevData(), outBiases->getDevData()};
	int myLen[numDataType] = {hidVisLen, hidBiasLen, avgOutLen, outBiasLen};

	#pragma omp parallel num_threads(transOPTimesInPro * (numProcess - 1)) 
	{

		MPI_Status status;
		int myState = 0;

		int tid = omp_get_thread_num();
		int pid = tid / transOPTimesInPro + 1;
		int swapId = tid % transOPTimesInPro;
		int dataAddr = tid % numDataType;

//		cout << "tid" << tid<< endl;

		while(myState != THREAD_END){
			MPI_Recv(&myState, 1, MPI_INT, pid, \
					swapId*10000, MPI_COMM_WORLD, &status);

			if(swapId < numDataType){
				MPI_Recv(myData[dataAddr], myLen[dataAddr], MPI_FLOAT, pid, \
						swapId+ myState, MPI_COMM_WORLD, &status);
			}else{
				MPI_Send(myData[dataAddr], myLen[dataAddr], MPI_FLOAT, pid, \
						swapId + myState, MPI_COMM_WORLD);
			}   
		}
	}

	delete nvTrainData;
	delete nvTrainLabel;
	delete nvValidData;
	delete nvValidLabel;
	delete hHidVis;
	delete hHidBiases;
	delete hAvgout;
	delete hOutBiases;
	delete hidVis;
	delete hidBiases;
	delete avgOut;
	delete outBiases;
}


void workerNode(pars* cnn, pars* logistic){
	int inLen = cnn->inSize * cnn->inSize * cnn->inChannel;
	int hidVisLen = cnn->numFilters * cnn->filterSize \
			* cnn->filterSize;
	int hidBiasLen = cnn->numFilters * 1;
	int avgOutLen = cnn->poolResultSize * cnn->poolResultSize * cnn->numFilters * logistic->numOut;
	int outBiasLen = logistic->numOut;
	int miniDataLen = cnn->minibatchSize * inLen;
	int miniLabelLen = cnn->minibatchSize;

	int proTrainDataLen = cnn->trainNum * inLen / (numProcess - 1);
	int proTrainLabelLen = cnn->trainNum / (numProcess - 1);
	int proValidDataLen = cnn->validNum * inLen / (numProcess - 1);
	int proValidLabelLen = cnn->validNum / (numProcess - 1);

	NVMatrix* nvTrainData = new NVMatrix(cnn->trainNum/(numProcess-1), inLen);
//				NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	NVMatrix* nvValidData = new NVMatrix(cnn->validNum/(numProcess-1), inLen);
//			NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	NVMatrix* nvTrainLabel = new NVMatrix(cnn->trainNum / (numProcess - 1), 1);
//				NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	NVMatrix* nvValidLabel = new NVMatrix(cnn->validNum / (numProcess - 1), 1);
//				NVMatrix::ALLOC_ON_UNIFIED_MEMORY);

	NVMatrix* miniData = new NVMatrix(nvTrainData->getDevData(), \
			cnn->minibatchSize, inLen);
	NVMatrix* miniLabel = new NVMatrix(nvTrainLabel->getDevData(), \
			cnn->minibatchSize, 1);

	Matrix* hHidVis = new Matrix(cnn->numFilters, cnn->filterSize * cnn->filterSize);
	Matrix* hHidBiases = new Matrix(cnn->numFilters, 1); 
	Matrix* hAvgout = new Matrix(avgOutLen / logistic->numOut, logistic->numOut);
	Matrix* hOutBiases = new Matrix(1, logistic->numOut);

	MPI_Bcast(hHidVis->getData(), hidVisLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(hHidBiases->getData(), hidBiasLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(hAvgout->getData(), avgOutLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(hOutBiases->getData(), outBiasLen, MPI_FLOAT, 0, MPI_COMM_WORLD);


	MPI_Status status;
	MPI_Recv(nvTrainData->getDevData(), proTrainDataLen, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);
	MPI_Recv(nvTrainLabel->getDevData(), proTrainLabelLen, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);
	MPI_Recv(nvValidData->getDevData(), proValidDataLen, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);
	MPI_Recv(nvValidLabel->getDevData(), proValidLabelLen, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);


	ConvNet layer1(hHidVis, hHidBiases, cnn);
	layer1.initCuda();
	Logistic layer2(hAvgout, hOutBiases, logistic);
	layer2.initCuda();

	int passMsg = 0;

	NVMatrix* y_i;
	NVMatrix* hidVis;
	NVMatrix* hidBiases;
	NVMatrix* avgOut;
	NVMatrix* outBiases;
	NVMatrix* dE_dy_j;

	clock_t t;
	t = clock();


	for(int epochIdx = 0; epochIdx < cnn->numEpoches; epochIdx++){
		int error = 0;

		for(int batchIdx = 0; batchIdx < cnn->numMinibatches; batchIdx++){

			miniData->changePtrFromStart(nvTrainData->getDevData(), \
					miniDataLen * batchIdx);
			miniLabel->changePtrFromStart(nvTrainLabel->getDevData(), \
					miniLabelLen * batchIdx);
			layer1.computeConvOutputs(miniData);
			layer1.computeMaxOutputs();
			y_i = layer1.getYI();
			layer2.computeClassOutputs(y_i);
			layer2.computeError(miniLabel, error);
			dE_dy_j = layer2.getDEDYJ();
			avgOut = layer2.getAvgOut();
			layer2.computeDerivs(y_i, miniLabel);
			layer1.computeDerivs(miniData, dE_dy_j, avgOut);
			layer1.updatePars();
			layer2.updatePars();
			avgOut = layer2.getAvgOut();
			outBiases = layer2.getOutBias();
			hidVis = layer1.getHidVis();
			hidBiases = layer1.getHidBias();
			if((batchIdx + 1) % cnn->nPush == 0){
				if(epochIdx == cnn->numEpoches - 1){
					if((batchIdx + cnn->nPush) >= cnn->numMinibatches \
							|| batchIdx == cnn->numMinibatches - 1)
						passMsg = THREAD_END;
					else
						passMsg = batchIdx;
				}
				else
					passMsg = batchIdx;
				MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_HIDVIS_PUSH*10000, \
						MPI_COMM_WORLD);
				MPI_Send(hidVis->getDevData(), hidVisLen, \
						MPI_FLOAT, 0, SWAP_HIDVIS_PUSH + passMsg, MPI_COMM_WORLD);

				MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_HIDBIAS_PUSH*10000, \
						MPI_COMM_WORLD);
				MPI_Send(hidBiases->getDevData(), hidBiasLen, \
						MPI_FLOAT, 0, SWAP_HIDBIAS_PUSH + passMsg, MPI_COMM_WORLD);

				MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_AVGOUT_PUSH*10000, \
						MPI_COMM_WORLD);
				MPI_Send(avgOut->getDevData(), avgOutLen, \
						MPI_FLOAT, 0, SWAP_AVGOUT_PUSH + passMsg, MPI_COMM_WORLD);

				MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_OUTBIAS_PUSH*10000, \
						MPI_COMM_WORLD);
				MPI_Send(outBiases->getDevData(), outBiasLen, \
						MPI_FLOAT, 0, SWAP_OUTBIAS_PUSH + passMsg, MPI_COMM_WORLD);
			}

			if((batchIdx + 1) % cnn->nFetch == 0){
				if(epochIdx == cnn->numEpoches - 1){
					if((batchIdx + cnn->nFetch) >= cnn->numMinibatches \
							|| batchIdx == cnn->numMinibatches - 1)
						passMsg = THREAD_END;
					else
						passMsg = batchIdx;
				}else
					passMsg = batchIdx;
				MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_HIDVIS_FETCH*10000, \
						MPI_COMM_WORLD);
				MPI_Recv(hidVis->getDevData(), hidVisLen, MPI_FLOAT, 0, \
						SWAP_HIDVIS_FETCH + passMsg, MPI_COMM_WORLD, &status);

				MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_HIDBIAS_FETCH*10000, \
						MPI_COMM_WORLD);
				MPI_Recv(hidBiases->getDevData(), hidBiasLen, MPI_FLOAT, \
						0, SWAP_HIDBIAS_FETCH + passMsg, \
						MPI_COMM_WORLD, &status);

				MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_AVGOUT_FETCH*10000, \
						MPI_COMM_WORLD);
				MPI_Recv(avgOut->getDevData(), avgOutLen, MPI_FLOAT, 0, \
						SWAP_AVGOUT_FETCH + passMsg, MPI_COMM_WORLD, &status);

				MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_OUTBIAS_FETCH*10000, \
						MPI_COMM_WORLD);
				MPI_Recv(outBiases->getDevData(), outBiasLen, MPI_FLOAT, \
						0, SWAP_OUTBIAS_FETCH + passMsg, \
						MPI_COMM_WORLD, &status);
			}
			if(batchIdx == cnn->numMinibatches - 1){ 
				int errorValid = 0;
				float loglihoodValid = 0;
				for(int validIdx = 0; validIdx < cnn->numValidBatches; validIdx++){

					miniData->changePtrFromStart(nvValidData->getDevData(), \
							miniDataLen * validIdx);
					miniLabel->changePtrFromStart(nvValidLabel->getDevData(), \
							miniLabelLen * validIdx);
					layer1.computeConvOutputs(miniData);
					layer1.computeMaxOutputs();
					y_i = layer1.getYI();
					layer2.computeClassOutputs(y_i);
					loglihoodValid += layer2.computeError(miniLabel, errorValid);

				}
				int totalValid = errorValid;
				if(numProcess > 2){
					if(rank == 1){
						for(int i = 2; i < numProcess; i++){
							MPI_Recv(&errorValid, 1, MPI_INT, i, i, \
									MPI_COMM_WORLD, &status);   
							totalValid += errorValid;
						}       
					}else{  
						MPI_Send(&errorValid, 1, MPI_INT, 1, rank, MPI_COMM_WORLD);
					}       
				}       
				if(rank == 1)
					cout << "epochIdx: " << epochIdx << ",error: " \
						<< (float)totalValid/cnn->validNum \
						<< ",likelihood: "<< loglihoodValid<< endl;
			}  

		}
		if(rank == 1){
			t = clock() - t;
			cout << " " << ((float)t/CLOCKS_PER_SEC) << " seconds.\n";
			t = clock();
		}

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

	if(numProcess <= 1){
		printf("Error: process number must bigger than 1\n");
		MPI_Abort(MPI_COMM_WORLD, 0); 
	}

	//检测有几个gpu
	int numGpus;
	cudaGetDeviceCount(&numGpus);
	cudaSetDevice(rank%numGpus);

/*
	// Ensure that RDMA ENABLED CUDA is set correctly
    int direct = getenv("MPICH_RDMA_ENABLED_CUDA")==NULL?0:atoi(getenv ("MPICH_RDMA_ENABLED_CUDA"));
    if(direct != 1){
        printf ("MPICH_RDMA_ENABLED_CUDA not enabled!\n");
        exit (EXIT_FAILURE);
    }
*/

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
	cnn->stepSize = 1;
	cnn->convResultSize = cnn->inSize - cnn->filterSize + 1;
	cnn->poolSize = 2;
	cnn->poolResultSize = cnn->convResultSize / cnn->poolSize;
	cnn->trainNum = 50000;
	cnn->validNum = 10000;
	cnn->minibatchSize = 100;
	cnn->numMinibatches = cnn->trainNum / (cnn->minibatchSize * (numProcess - 1));
	cnn->numValidBatches = cnn->validNum / (cnn->minibatchSize * (numProcess - 1));
	cnn->numEpoches = 1; 
	cnn->nPush = 4;
	cnn->nFetch = 5;

	logistic->wcAvgOut = 0;
	logistic->epsAvgOut = 0.1;
	logistic->epsOutBias = 0.1;
	logistic->mom = 0;
	logistic->numOut = 10; 
	logistic->minibatchSize = 100;

	if(rank == 0){ 
		managerNode(cnn, logistic);
	}   
	else{
		workerNode(cnn, logistic);
	} 	

	delete cnn;
	delete logistic;
	MPI_Finalize();
	return 0;
}




















