/*
 * filename:main.cu
 */

#include <iostream>
#include <fstream>
#include <time.h>
#include <sstream>
#include <cmath>
#include <omp.h>
#include "mpi.h"
#include "matrix.h"
#include "nvmatrix.cuh"
#include "convnet.cuh"
#include "convnet_kernel.cuh"
#include "utils.h"
#include "logistic.cuh"
#include "load_layer.hpp"

using namespace std;

#define THREAD_END 100000
enum swapInfo{SWAP_HIDVIS1_PUSH, SWAP_HIDBIAS1_PUSH, \
	SWAP_HIDVIS2_PUSH, SWAP_HIDBIAS2_PUSH,
	SWAP_AVGOUT_PUSH, SWAP_OUTBIAS_PUSH,
	SWAP_HIDVIS1_FETCH, SWAP_HIDBIAS1_FETCH, \
	SWAP_HIDVIS2_FETCH, SWAP_HIDBIAS2_FETCH, \
		SWAP_AVGOUT_FETCH, SWAP_OUTBIAS_FETCH};

int numProcess;
int rank;

void managerNode(pars* cnnPars, pars* logistic){

	cout << "=========================\n" \
		<< "train: " << cnnPars[0].trainNum \
		<< "\nvalid: " << cnnPars[0].validNum \
		<< "\nfiltersize: " << cnnPars[0].filterSize \
		<< "\nnumFilters: " << cnnPars[0].numFilters \
		<< "\nconvstepsize: " << cnnPars[0].stepSize \
		<< "\nepsHidVis: " << cnnPars[0].epsHidVis \
		<< "\nepsHidBias: " << cnnPars[0].epsHidBias \
		<< "\nepsAvgOut: " << logistic->epsAvgOut \
		<< "\nepsOutBias: " << logistic->epsOutBias \
		<< "\nmom: " << cnnPars[0].mom \
		<< "\nwcHidVis: " << cnnPars[0].wcHidVis \
		<< "\nwcAvgOut: " << logistic->wcAvgOut << endl;

	int cnn1InLen = cnnPars[0].inSize * cnnPars[0].inSize * cnnPars[0].inChannel;
	int cnn1HidVisLen = cnnPars[0].numFilters * cnnPars[0].filterSize \
			* cnnPars[0].filterSize;
	int cnn1HidBiasLen = cnnPars[0].numFilters * 1;

	int cnn2HidVisLen = cnnPars[1].numFilters * cnnPars[1].filterSize \
			* cnnPars[1].filterSize;
	int cnn2HidBiasLen = cnnPars[1].numFilters * 1;

	int avgOutLen = cnnPars[0].poolResultSize * cnnPars[0].poolResultSize * cnnPars[0].numFilters * logistic->numOut;
//	int avgOutLen = cnn1.inSize * cnn1.inSize * cnn1.inChannel * logistic->numOut;
	int outBiasLen = logistic->numOut;

	int proTrainDataLen = cnnPars[0].trainNum * cnn1InLen / (numProcess - 1);
	int proTrainLabelLen = cnnPars[0].trainNum / (numProcess - 1);
	int proValidDataLen = cnnPars[0].validNum * cnn1InLen / (numProcess - 1);
	int proValidLabelLen = cnnPars[0].validNum / (numProcess - 1);

	NVMatrix* nvTrainData = new NVMatrix(cnnPars[0].trainNum, cnn1InLen);
	NVMatrix* nvValidData = new NVMatrix(cnnPars[0].validNum, cnn1InLen);
	NVMatrix* nvTrainLabel = new NVMatrix(cnnPars[0].trainNum, 1);
	NVMatrix* nvValidLabel = new NVMatrix(cnnPars[0].validNum, 1);

    readData(nvTrainData, "./data/input/mnist_train.bin", true);
    readData(nvValidData, "./data/input/mnist_valid.bin", true);
    readData(nvTrainLabel, "./data/input/mnist_label_train.bin", false);
    readData(nvValidLabel, "./data/input/mnist_label_valid.bin", false);


	ImgInfo<float> *cifar10Info = new ImgInfo<float>;
	LoadCifar10<float> cifar10(cifar10Info);
/*
    for(int i = 1; i < 6; i++){
        string s;
        stringstream ss;
        ss << 5;
        ss >> s;    
		string filename = "./data/cifar-10-batches-bin/data_batch_"+s+".bin";
        cifar10.loadBinary(filename, cifar10Info->train_pixel_ptr, \
				cifar10Info->train_label_ptr);    
    }   
    cifar10.loadBinary("./data/cifar-10-batches-bin/test_batch.bin", \
            cifar10Info->test_pixel_ptr, cifar10Info->test_label_ptr);

	nvTrainData->copyFromHost(cifar10Info->train_pixel, cnnPars[0].trainNum * cnn1InLen);
	nvTrainLabel->copyFromHost(cifar10Info->train_label, cnnPars[0].trainNum);
	nvValidData->copyFromHost(cifar10Info->test_pixel, cnnPars[0].validNum * cnn1InLen);
	nvValidLabel->copyFromHost(cifar10Info->test_label, cnnPars[0].validNum);
*/
	NVMatrix* cnn1HidVis = new NVMatrix(cnnPars[0].numFilters, \
			cnnPars[0].filterSize * cnnPars[0].filterSize);
	NVMatrix* cnn1HidBiases = new NVMatrix(cnnPars[0].numFilters, 1);
	NVMatrix* avgOut = new NVMatrix(avgOutLen / logistic->numOut, logistic->numOut);
	NVMatrix* outBiases = new NVMatrix(1, logistic->numOut);

	NVMatrix* cnn2HidVis = new NVMatrix(cnnPars[1].numFilters, \
			cnnPars[1].filterSize * cnnPars[1].filterSize);
	NVMatrix* cnn2HidBiases = new NVMatrix(cnnPars[1].numFilters, 1);

	initW(cnn1HidVis);
	initW(cnn2HidVis);
	cudaMemset(cnn1HidBiases->getDevData(), 0, sizeof(float) * cnnPars[0].numFilters);
	cudaMemset(cnn2HidBiases->getDevData(), 0, sizeof(float) * cnnPars[1].numFilters);
	cudaMemset(avgOut->getDevData(), 0, sizeof(float) * avgOutLen);
	cudaMemset(outBiases->getDevData(), 0, sizeof(float) * logistic->numOut);

	//	readPars(hHidVis, "hHidVis_t1.bin");
	//	readPars(hHidBiases, "hHidBiases_t1.bin");
	//	readPars(hAvgout, "hAvgout_t1.bin");
	//	readPars(hOutBiases, "hOutBiases_t1.bin");

	MPI_Bcast(cnn1HidVis->getDevData(), cnn1HidVisLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn1HidBiases->getDevData(), cnn1HidBiasLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn2HidVis->getDevData(), cnn2HidVisLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn2HidBiases->getDevData(), cnn2HidBiasLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(avgOut->getDevData(), avgOutLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(outBiases->getDevData(), outBiasLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
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
	const int transOPTimesInPro = 12;
	const int numDataType = 6;
	float* myData[numDataType] = {cnn1HidVis->getDevData(), cnn1HidBiases->getDevData(), \
			cnn2HidVis->getDevData(), cnn2HidBiases->getDevData(), \
			avgOut->getDevData(), outBiases->getDevData()};
	int myLen[numDataType] = {cnn1HidVisLen, cnn1HidBiasLen, cnn2HidVisLen, cnn2HidBiasLen, \
				avgOutLen, outBiasLen};

	#pragma omp parallel num_threads(transOPTimesInPro * (numProcess - 1)) 
	{

		MPI_Status status;
		int myState = 0;

		int tid = omp_get_thread_num();
		int pid = tid / transOPTimesInPro + 1;
		int swapId = tid % transOPTimesInPro;
		int dataAddr = tid % numDataType;

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

	delete cifar10Info;
	delete nvTrainData;
	delete nvTrainLabel;
	delete nvValidData;
	delete nvValidLabel;
	delete cnn1HidVis;
	delete cnn1HidBiases;
	delete avgOut;
	delete outBiases;
}


void workerNode(pars* cnnPars, pars* logistic){
	
	cnnPars->trainNum /= (numProcess - 1);
	int cnn1InLen = cnnPars->inSize * cnnPars->inSize * cnnPars->inChannel;
	int cnn1HidVisLen = cnnPars->numFilters * cnnPars->filterSize \
			* cnnPars->filterSize;
	int cnn1HidBiasLen = cnnPars->numFilters * 1;
	int cnn2HidVisLen = cnnPars[1].numFilters * cnnPars[1].filterSize \
			* cnnPars[1].filterSize;
	int cnn2HidBiasLen = cnnPars[1].numFilters * 1;
	int avgOutLen = cnnPars->poolResultSize * cnnPars->poolResultSize * cnnPars->numFilters * logistic->numOut;
//	int avgOutLen = cnnPars->inSize * cnn1->inSize * cnn1->inChannel * logistic->numOut;
	int outBiasLen = logistic->numOut;


	int miniDataLen = cnnPars->minibatchSize * cnn1InLen;
	int miniLabelLen = cnnPars->minibatchSize;

	int proTrainDataLen = cnnPars->trainNum * cnn1InLen;
	int proTrainLabelLen = cnnPars->trainNum;
	int proValidDataLen = cnnPars->validNum * cnn1InLen;
	int proValidLabelLen = cnnPars->validNum;

	ConvNet cnn1(cnnPars);
	cnn1.initCuda();

	ConvNet cnn2(cnnPars+1);
	cnn2.initCuda();

	Logistic layer3(logistic);
	layer3.initCuda();
	NVMatrix* cnn1HidVis = cnn1.getHidVis();
	NVMatrix* cnn1HidBiases = cnn1.getHidBias();
	NVMatrix* cnn2HidVis = cnn2.getHidVis();
	NVMatrix* cnn2HidBiases = cnn2.getHidBias();
	NVMatrix* avgOut = layer3.getAvgOut();
	NVMatrix* outBiases = layer3.getOutBias();

	MPI_Bcast(cnn1HidVis->getDevData(), cnn1HidVisLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn1HidBiases->getDevData(), cnn1HidBiasLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn2HidVis->getDevData(), cnn2HidVisLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn2HidBiases->getDevData(), cnn2HidBiasLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(avgOut->getDevData(), avgOutLen, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(outBiases->getDevData(), outBiasLen, MPI_FLOAT, 0, MPI_COMM_WORLD);

	NVMatrix* cnn1TrainData = cnn1.getTrainData();
	NVMatrix* cnn1TrainLabel = cnn1.getTrainLabel();
	NVMatrix* cnn1ValidData = cnn1.getValidData();
	NVMatrix* cnn1ValidLabel = cnn1.getValidLabel();

	NVMatrix* miniData = new NVMatrix(cnnPars->minibatchSize, cnn1InLen);
	NVMatrix* miniLabel = new NVMatrix(cnnPars->minibatchSize, 1);

	MPI_Status status;
	MPI_Recv(cnn1TrainData->getDevData(), proTrainDataLen, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);
	MPI_Recv(cnn1TrainLabel->getDevData(), proTrainLabelLen, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);
	MPI_Recv(cnn1ValidData->getDevData(), proValidDataLen, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);
	MPI_Recv(cnn1ValidLabel->getDevData(), proValidLabelLen, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);

	int passMsg = 0;

	NVMatrix* y_i;
	NVMatrix* dE_dy_i;

	clock_t t;
	t = clock();

	for(int epochIdx = 0; epochIdx < cnnPars->numEpoches; epochIdx++){
		int error = 0;

		for(int batchIdx = 0; batchIdx < cnnPars->numMinibatches; batchIdx++){

			miniData->changePtrFromStart(cnn1TrainData->getDevData(), \
					miniDataLen * batchIdx);
			miniLabel->changePtrFromStart(cnn1TrainLabel->getDevData(), \
					miniLabelLen * batchIdx);
			cnn1.computeConvOutputs(miniData);
			cnn1.computeMaxOutputs();
			y_i = cnn1.getYI();
			dE_dy_i = cnn1.getDEDYI();
			layer3.computeClassOutputs(y_i);
//			layer3.computeClassOutputs(miniData);
			layer3.computeError(miniLabel, error);
//cout << "done2\n";
			avgOut = layer3.getAvgOut();
//			layer3.computeDerivs(miniData, miniLabel);
			layer3.computeDerivs(y_i, miniLabel, dE_dy_i);
			cnn1.computeDerivs(miniData);
			cnn1.updatePars();
			layer3.updatePars();
			avgOut = layer3.getAvgOut();
			outBiases = layer3.getOutBias();
			cnn1HidVis = cnn1.getHidVis();
			cnn1HidBiases = cnn1.getHidBias();
			if((batchIdx + 1) % cnnPars->nPush == 0){
				if(epochIdx == cnnPars->numEpoches - 1){
					if((batchIdx + cnnPars->nPush) >= cnnPars->numMinibatches \
							|| batchIdx == cnnPars->numMinibatches - 1)
						passMsg = THREAD_END;
					else
						passMsg = batchIdx;
				}
				else
					passMsg = batchIdx;
				/*
				const int numDataType = 4;
				float* myData[numDataType] = {cnn1HidVis->getDevData(), cnn1HidBiases->getDevData(), \
						avgOut->getDevData(), outBiases->getDevData()};
			    int myLen[numDataType] = {cnn1HidVisLen, cnn1HidBiasLen, avgOutLen, outBiasLen};
				#pragma omp parallel num_threads(numDataType)
				{
					int tid = omp_get_thread_num();
					int dataAddr = tid % numDataType;
					int swapId = tid % numDataType;

					MPI_Recv(&passMsg, 1, MPI_INT, 0, \
						swapId*10000, MPI_COMM_WORLD, &status);
					MPI_Send(myData[dataAddr], myLen[dataAddr], \
						MPI_FLOAT, 0, swapId + passMsg, MPI_COMM_WORLD);
					
				}*/
				MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_HIDVIS1_PUSH*10000, \
                        MPI_COMM_WORLD);
                MPI_Send(cnn1HidVis->getDevData(), cnn1HidVisLen, \
                        MPI_FLOAT, 0, SWAP_HIDVIS1_PUSH + passMsg, MPI_COMM_WORLD);

                MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_HIDBIAS1_PUSH*10000, \
                        MPI_COMM_WORLD);
                MPI_Send(cnn1HidBiases->getDevData(), cnn1HidBiasLen, \
                        MPI_FLOAT, 0, SWAP_HIDBIAS1_PUSH + passMsg, MPI_COMM_WORLD);

				MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_HIDVIS2_PUSH*10000, \
                        MPI_COMM_WORLD);
                MPI_Send(cnn2HidVis->getDevData(), cnn2HidVisLen, \
                        MPI_FLOAT, 0, SWAP_HIDVIS2_PUSH + passMsg, MPI_COMM_WORLD);

                MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_HIDBIAS2_PUSH*10000, \
                        MPI_COMM_WORLD);
                MPI_Send(cnn2HidBiases->getDevData(), cnn2HidBiasLen, \
                        MPI_FLOAT, 0, SWAP_HIDBIAS2_PUSH + passMsg, MPI_COMM_WORLD);

                MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_AVGOUT_PUSH*10000, \
                        MPI_COMM_WORLD);
                MPI_Send(avgOut->getDevData(), avgOutLen, \
                        MPI_FLOAT, 0, SWAP_AVGOUT_PUSH + passMsg, MPI_COMM_WORLD);

                MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_OUTBIAS_PUSH*10000, \
                        MPI_COMM_WORLD);
                MPI_Send(outBiases->getDevData(), outBiasLen, \
                        MPI_FLOAT, 0, SWAP_OUTBIAS_PUSH + passMsg, MPI_COMM_WORLD);

			}
			if((batchIdx + 1) % cnnPars->nFetch == 0){
				if(epochIdx == cnnPars->numEpoches - 1){
					if((batchIdx + cnnPars->nFetch) >= cnnPars->numMinibatches \
							|| batchIdx == cnnPars->numMinibatches - 1)
						passMsg = THREAD_END;
					else
						passMsg = batchIdx;
				}else
					passMsg = batchIdx;
			
				MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_HIDVIS1_FETCH*10000, \
						MPI_COMM_WORLD);
				MPI_Recv(cnn1HidVis->getDevData(), cnn1HidVisLen, MPI_FLOAT, 0, \
						SWAP_HIDVIS1_FETCH + passMsg, MPI_COMM_WORLD, &status);

				MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_HIDBIAS1_FETCH*10000, \
						MPI_COMM_WORLD);
				MPI_Recv(cnn1HidBiases->getDevData(), cnn1HidBiasLen, MPI_FLOAT, \
						0, SWAP_HIDBIAS1_FETCH + passMsg, \
						MPI_COMM_WORLD, &status);

				MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_HIDVIS2_FETCH*10000, \
						MPI_COMM_WORLD);
				MPI_Recv(cnn2HidVis->getDevData(), cnn2HidVisLen, MPI_FLOAT, 0, \
						SWAP_HIDVIS2_FETCH + passMsg, MPI_COMM_WORLD, &status);

				MPI_Send(&passMsg, 1, MPI_INT, 0, SWAP_HIDBIAS2_FETCH*10000, \
						MPI_COMM_WORLD);
				MPI_Recv(cnn2HidBiases->getDevData(), cnn2HidBiasLen, MPI_FLOAT, \
						0, SWAP_HIDBIAS2_FETCH + passMsg, \
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


			if(batchIdx == cnnPars->numMinibatches - 1){ 
				int errorValid = 0;
				float loglihoodValid = 0;
				for(int validIdx = 0; validIdx < cnnPars->numValidBatches; validIdx++){

					miniData->changePtrFromStart(cnn1ValidData->getDevData(), \
							miniDataLen * validIdx);
					miniLabel->changePtrFromStart(cnn1ValidLabel->getDevData(), \
							miniLabelLen * validIdx);
					cnn1.computeConvOutputs(miniData);
					cnn1.computeMaxOutputs();
					y_i = cnn1.getYI();
					layer3.computeClassOutputs(y_i);
//					layer3.computeClassOutputs(miniData);
					loglihoodValid += layer3.computeError(miniLabel, errorValid);

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
						<< (float)totalValid/cnnPars->validNum \
						<< ",likelihood: "<< loglihoodValid<< endl;
			}
		}
	//	if((epochIdx + 1) % 10 == 0){
	//		cnn1.transfarLowerPars();
	//		layer3.transfarLowerPars();
	//	}  
		if(rank == 1){
			t = clock() - t;
			cout << " " << ((float)t/CLOCKS_PER_SEC) << " seconds.\n";
			t = clock();
		}

	}
	delete miniData;
	delete miniLabel;
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


	const int numCnnLayers = 2;
	
	pars* cnn1Pars = new pars[numCnnLayers];
	pars* logistic = new pars;

	cnn1Pars[0].epsHidVis = 0.1;
	cnn1Pars[0].epsHidBias = 0.1;
	cnn1Pars[0].mom = 0.9;
	cnn1Pars[0].wcHidVis = 0;
	cnn1Pars[0].inSize = 28; 
	cnn1Pars[0].inChannel = 1;
	cnn1Pars[0].filterSize = 5;
	cnn1Pars[0].numFilters = 16; 
	cnn1Pars[0].stepSize = 1;
	cnn1Pars[0].convResultSize = (cnn1Pars[0].inSize - cnn1Pars[0].filterSize) / cnn1Pars[0].stepSize + 1;
	cnn1Pars[0].poolSize = 2;
	cnn1Pars[0].poolResultSize = cnn1Pars[0].convResultSize / cnn1Pars[0].poolSize;
	cnn1Pars[0].trainNum = 50000;
	cnn1Pars[0].validNum = 10000;
	cnn1Pars[0].minibatchSize = 100;
	cnn1Pars[0].numMinibatches = cnn1Pars[0].trainNum / (cnn1Pars[0].minibatchSize * (numProcess - 1));
	cnn1Pars[0].numValidBatches = cnn1Pars[0].validNum / (cnn1Pars[0].minibatchSize * (numProcess - 1));
	cnn1Pars[0].numEpoches = 100; 
	cnn1Pars[0].nPush = 1;
	cnn1Pars[0].nFetch = 1;
	cnn1Pars[0].finePars = 0.995;

	cnn1Pars[1].epsHidVis = 0.01;
	cnn1Pars[1].epsHidBias = 0.01;
	cnn1Pars[1].mom = 0.9;
	cnn1Pars[1].wcHidVis = 0;
	cnn1Pars[1].inSize = cnn1Pars[0].poolResultSize; 
	cnn1Pars[1].inChannel = cnn1Pars[0].numFilters;
	cnn1Pars[1].filterSize = 3;
	cnn1Pars[1].numFilters = 128; 
	cnn1Pars[1].stepSize = 1;
	cnn1Pars[1].convResultSize = (cnn1Pars[1].inSize - cnn1Pars[1].filterSize) / cnn1Pars[1].stepSize + 1;
	cnn1Pars[1].poolSize = 2;
	cnn1Pars[1].poolResultSize = cnn1Pars[1].convResultSize / cnn1Pars[1].poolSize;
	cnn1Pars[1].trainNum = 50000;
	cnn1Pars[1].validNum = 10000;
	cnn1Pars[1].minibatchSize = 100;
	cnn1Pars[1].numMinibatches = cnn1Pars[1].trainNum / (cnn1Pars[1].minibatchSize * (numProcess - 1));
	cnn1Pars[1].numValidBatches = cnn1Pars[1].validNum / (cnn1Pars[1].minibatchSize * (numProcess - 1));
	cnn1Pars[1].numEpoches = 100; 
	cnn1Pars[1].nPush = 1;
	cnn1Pars[1].nFetch = 1;

	logistic->wcAvgOut = 0;
	logistic->epsAvgOut = 0.001;
	logistic->epsOutBias = 0.001;
	logistic->numIn = 12*12*16;
	logistic->mom = 0.9;
	logistic->numOut = 10; 
	logistic->minibatchSize = 100;
	logistic->finePars = 0.995;

	if(rank == 0){ 
		managerNode(cnn1Pars, logistic);
	}   
	else{
		workerNode(cnn1Pars, logistic);
	} 	

	delete[] cnn1Pars;
	delete logistic;
	MPI_Finalize();
	return 0;
}




















