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
		<< "\nepsAvgOut: " << cnn->epsAvgOut \
		<< "\nepsOutBias: " << cnn->epsOutBias \
		<< "\nmom: " << cnn->mom \
		<< "\nwcHidVis: " << cnn->wcHidVis \
		<< "\nwcAvgOut: " << cnn->wcAvgOut << endl;

	int inLen = cnn->inSize * cnn->inSize * cnn->inChannel;
	int hidVisLen = cnn->numFilters * cnn->filterSize \
					* cnn->filterSize;
	int hidBiasLen = cnn->numFilters * 1;
	int avgOutLen = cnn->convResultSize * cnn->convResultSize * cnn->numFilters * logistic->numOut;
	int outBiasLen = logistic->numOut;

	int proTrainDataLen = cnn->trainNum * inLen / (numProcess - 1);
	int proTrainLabelLen = cnn->trainNum / (numProcess - 1);
	int proValidDataLen = cnn->validNum * inLen / (numProcess - 1);
	int proValidLabelLen = cnn->validNum / (numProcess - 1);

	NVMatrix* nvTrainData = new NVMatrix(cnn->trainNum, inLen);
	NVMatrix* nvValidData = new NVMatrix(cnn->validNum, inLen);
	NVMatrix* nvTrainLabel = new NVMatrix(cnn->trainNum, 1);
	NVMatrix* nvValidLabel = new NVMatrix(cnn->validNum, 1);

	readData(nvTrainData, "../data/input/mnist_train.bin", true);
	readData(nvValidData, "../data/input/mnist_valid.bin", true);
	readData(nvTrainLabel, "../data/input/mnist_label_train.bin", false);
	readData(nvValidLabel, "../data/input/mnist_label_valid.bin", false);

	Matrix* hHidVis = new Matrix(cnn->numFilters, cnn->filterSize * cnn->filterSize);
	Matrix* hHidBiases = new Matrix(cnn->numFilters, 1);
	Matrix* hAvgout = new Matrix(avgOutLen, logistic->numOut);
	Matrix* hOutBiases = new Matrix(1, logistic->numOut);

	initW(hHidVis->getData(), cnn->numFilters * cnn->filterSize * cnn->filterSize);
	memset(hHidBiases->getData(), 0, sizeof(float) * cnn->numFilters);
	memset(hAvgout->getData(), 0, sizeof(float) * avgOutLen * logistic->numOut);
	memset(hOutBiases->getData(), 0, sizeof(float) * logistic->numOut);

	//	readPars(hHidVis, "hHidVis_t1.bin");
	//	readPars(hHidBiases, "hHidBiases_t1.bin");
	//	readPars(hAvgout, "hAvgout_t1.bin");
	//	readPars(hOutBiases, "hOutBiases_t1.bin");

	MPI_Bcast(hHidVis->getData(), hidVisLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(hHidBiases->getData(), hidBiasLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(hAvgout->getData(), avgOutLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(hOutBiases->getData(), outBiasLen, MPI_FLOAT, 0, MPI_COMM_WORLD);

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


	clock_t t;
	t = clock();
}

void workerNode(pars* cnn, pars* logistic){
	int inLen = cnn->inSize * cnn->inSize * cnn->inChannel;
	int convResultSize = cnn->inSize - cnn->filterSize + 1;
	int hidVisLen = cnn->numFilters * cnn->filterSize \
					* cnn->filterSize;
	int hidBiasLen = cnn->numFilters * 1;
	int avgOutLen = cnn->convResultSize * cnn->convResultSize * cnn->numFilters * logistic->numOut;
	int outBiasLen = logistic->numOut;
	int miniDataLen = logistic->minibatchSize * inLen;
	int miniLabelLen = logistic->minibatchSize;


	int proTrainDataLen = cnn->trainNum * inLen / (numProcess - 1);
	int proTrainLabelLen = cnn->trainNum / (numProcess - 1);
	int proValidDataLen = cnn->validNum * inLen / (numProcess - 1);
	int proValidLabelLen = cnn->validNum / (numProcess - 1);

	NVMatrix* nvTrainData = new NVMatrix(logistic->trainNum / (numProcess - 1), \
			inLen);
	NVMatrix* nvValidData = new NVMatrix(logistic->validNum / (numProcess - 1), \
			inLen);
	NVMatrix* nvTrainLabel = new NVMatrix(logistic->trainNum / (numProcess - 1), 1);
	NVMatrix* nvValidLabel = new NVMatrix(logistic->validNum / (numProcess - 1), 1);

	NVMatrix* miniData = new NVMatrix(nvTrainData->getDevData(), \
			logistic->minibatchSize, inLen);
	NVMatrix* miniLabel = new NVMatrix(nvTrainLabel->getDevData(), \
			logistic->minibatchSize, 1);

	Matrix* hHidVis = new Matrix(cnn->numFilters, cnn->filterSize * cnn->filterSize);
	Matrix* hHidBiases = new Matrix(cnn->numFilters, 1); 
	Matrix* hAvgout = new Matrix(avgOutLen, logistic->numOut);
	Matrix* hOutBiases = new Matrix(1, logistic->numOut);

	MPI_Bcast(hHidVis->getData(), hidVisLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(hHidBiases->getData(), hidBiasLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(hAvgout->getData(), avgOutLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(hOutBiases->getData(), outBiasLen, MPI_FLOAT, 0, MPI_COMM_WORLD);

	MPI_Status status;
cout << "???";
	MPI_Recv(nvTrainData->getDevData(), proTrainDataLen, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);
cout << "???1";
	MPI_Recv(nvTrainLabel->getDevData(), proTrainLabelLen, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);
cout << "???2";
	MPI_Recv(nvValidData->getDevData(), proValidDataLen, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);
cout << "???3";
	MPI_Recv(nvValidLabel->getDevData(), proValidLabelLen, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);

cout << "???4";

	ConvNet layer1(hHidVis, hHidBiases, cnn);
	layer1.initCuda();
	Logistic layer2(hAvgout, hOutBiases, logistic);
	layer2.initCuda();

	int passMsg;

	NVMatrix* y_i;
	NVMatrix* avgOut;
	NVMatrix* outBiases;
	NVMatrix* dE_dy_j;

	clock_t t;
	t = clock();

	for(int epochIdx = 0; epochIdx < logistic->numEpoches; epochIdx++){
		int error = 0;

		for(int batchIdx = 0; batchIdx < logistic->numMinibatches; batchIdx++){

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
			/*
			   if((batchIdx + 1) % logistic->nPush == 0){
			   if(epochIdx == logistic->numEpoches - 1 \
			   && ((batchIdx + logistic->nPush) > logistic->numMinibatches))
			   passMsg = THREAD_END;
			   else
			   passMsg = batchIdx;
			   MPI_Send(&passMsg, 1, MPI_INT, 0, \
			   SWAP_AVGOUT_PUSH*100, \
			   MPI_COMM_WORLD);
			   MPI_Send(avgOut->getDevData(), avgOutLen, \
			   MPI_FLOAT, 0, SWAP_AVGOUT_PUSH * 200 + passMsg, \
			   MPI_COMM_WORLD);
			   MPI_Send(&passMsg, 1, MPI_INT, 0, \
			   SWAP_BIAS_PUSH*100, \
			   MPI_COMM_WORLD);
			   MPI_Send(outBiases->getDevData(), outBiasLen, \
			   MPI_FLOAT, 0, SWAP_BIAS_PUSH*200 + passMsg, \
			   MPI_COMM_WORLD);
			   }

			   if((batchIdx + 1) % logistic->nFetch == 0){
			   if((epochIdx == logistic->numEpoches - 1) && \
			   ((batchIdx + logistic->nFetch) > logistic->numMinibatches)){
			   passMsg = THREAD_END;
			   }else
			   passMsg = batchIdx;
			   MPI_Send(&passMsg, 1, MPI_INT, 0, \
			   SWAP_AVGOUT_FETCH*100, \
			   MPI_COMM_WORLD);
			   MPI_Recv(avgOut->getDevData(), avgOutLen, MPI_FLOAT, 0, \
			   SWAP_AVGOUT_FETCH*200 + passMsg, \
			   MPI_COMM_WORLD, &status);

			   MPI_Send(&passMsg, 1, MPI_INT, 0, \
			   SWAP_BIAS_FETCH*100, \
			   MPI_COMM_WORLD);
			   MPI_Recv(outBiases->getDevData(), outBiasLen, MPI_FLOAT, \
			   0, SWAP_BIAS_FETCH*200 + passMsg, \
			   MPI_COMM_WORLD, &status);
			   }

			 */
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

	//检测有几个gpu
	int numGpus;
	cudaGetDeviceCount(&numGpus);
	cudaSetDevice(rank%numGpus);

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
	cnn->convResultSize = cnn->inSize - cnn->filterSize + 1;
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

	if(rank == 0){ 
		managerNode(cnn, logistic);
	}   
	else{
		workerNode(cnn, logistic);
	} 	

	delete cnn;
	delete logistic;
	return 0;
}




















