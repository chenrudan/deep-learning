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
#include "utils.cuh"
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

	cout << "\n===========overall==============" \
		<< "\ntrain: " << cnnPars[0].trainNum \
		<< "\nvalid: " << cnnPars[0].validNum \
		<< "\nbatchSize: " << cnnPars[0].minibatchSize \
		<< "\nnFetch: " << cnnPars[0].nFetch \
		<< "\nnPush: " << cnnPars[0].nPush;

	cout << "\n===========cnn1==============" \
		<< "\ninSize: " << cnnPars[0].inSize \
		<< "\ninChannel: " << cnnPars[0].inChannel \
		<< "\nfiltersize: " << cnnPars[0].filterSize \
		<< "\nnumFilters: " << cnnPars[0].numFilters \
		<< "\nconvstepsize: " << cnnPars[0].stepSize \
		<< "\npoolSize: " << cnnPars[0].poolSize \
		<< "\nepsHidVis: " << cnnPars[0].epsHidVis \
		<< "\nepsHidBias: " << cnnPars[0].epsHidBias \
		<< "\nmom: " << cnnPars[0].mom \
		<< "\nwcHidVis: " << cnnPars[0].wcHidVis;
		
	cout << "\n===========cnn2==============" \
		<< "\ninSize: " << cnnPars[1].inSize \
		<< "\ninChannel: " << cnnPars[1].inChannel \
		<< "\nfiltersize: " << cnnPars[1].filterSize \
		<< "\nnumFilters: " << cnnPars[1].numFilters \
		<< "\nconvstepsize: " << cnnPars[1].stepSize \
		<< "\npoolSize: " << cnnPars[1].poolSize \
		<< "\nepsHidVis: " << cnnPars[1].epsHidVis \
		<< "\nepsHidBias: " << cnnPars[1].epsHidBias \
		<< "\nmom: " << cnnPars[1].mom \
		<< "\nwcHidVis: " << cnnPars[1].wcHidVis ;

	cout << "\n===========logsitic==============" \
		<< "\ninSize: " << logistic->numIn \
		<< "\noutSize: " << logistic->numOut \
		<< "\nepsAvgOut: " << logistic->epsAvgOut \
		<< "\nepsOutBias: " << logistic->epsOutBias \
		<< "\nmom: " << logistic->mom \
		<< "\nwcAvgOut: " << logistic->wcAvgOut << endl;

	int cnn1InLen = cnnPars[0].inSize * cnnPars[0].inSize * cnnPars[0].inChannel;
	int cnn1HidVisLen = cnnPars[0].numFilters * cnnPars[0].filterSize \
			* cnnPars[0].filterSize * cnnPars[0].inChannel;
	int cnn1HidBiasLen = cnnPars[0].numFilters;

	int cnn2HidVisLen = cnnPars[1].numFilters * cnnPars[1].filterSize \
			* cnnPars[1].filterSize * cnnPars[1].inChannel;
	int cnn2HidBiasLen = cnnPars[1].numFilters;

	int avgOutLen = cnnPars[1].poolResultSize*cnnPars[1].poolResultSize*cnnPars[1].numFilters*logistic->numOut;
//	int avgOutLen = cnnPars[0].inSize * cnnPars[0].inSize * cnnPars[0].inChannel* logistic->numOut;
	int outBiasLen = logistic->numOut;

	int proTrainDataLen = cnnPars[0].trainNum * cnn1InLen / (numProcess - 1);
	int proTrainLabelLen = cnnPars[0].trainNum / (numProcess - 1);
	int proValidDataLen = cnnPars[0].validNum * cnn1InLen / (numProcess - 1);
	int proValidLabelLen = cnnPars[0].validNum / (numProcess - 1);

	NVMatrix* nvTrainData = new NVMatrix(cnnPars[0].trainNum, cnn1InLen);
	NVMatrix* nvValidData = new NVMatrix(cnnPars[0].validNum, cnn1InLen);
	NVMatrix* nvTrainLabel = new NVMatrix(cnnPars[0].trainNum, 1);
	NVMatrix* nvValidLabel = new NVMatrix(cnnPars[0].validNum, 1);

    readData(nvTrainData, "../data/input/mnist_train.bin", true);
    readData(nvValidData, "../data/input/mnist_valid.bin", true);
    readData(nvTrainLabel, "../data/input/mnist_label_train.bin", false);
    readData(nvValidLabel, "../data/input/mnist_label_valid.bin", false);


	ImgInfo<float> *cifar10Info = new ImgInfo<float>;
	LoadCifar10<float> cifar10(cifar10Info);
/*
    for(int i = 1; i < 6; i++){
        string s;
        stringstream ss;
        ss << 5;
        ss >> s;    
		string filename = "../data/cifar-10-batches-bin/data_batch_"+s+".bin";
        cifar10.loadBinary(filename, cifar10Info->train_pixel_ptr, \
				cifar10Info->train_label_ptr);    
    }   
    cifar10.loadBinary("../data/cifar-10-batches-bin/test_batch.bin", \
            cifar10Info->test_pixel_ptr, cifar10Info->test_label_ptr);

	nvTrainData->copyFromHost(cifar10Info->train_pixel, cnnPars[0].trainNum * cnn1InLen);
	nvTrainLabel->copyFromHost(cifar10Info->train_label, cnnPars[0].trainNum);
	nvValidData->copyFromHost(cifar10Info->test_pixel, cnnPars[0].validNum * cnn1InLen);
	nvValidLabel->copyFromHost(cifar10Info->test_label, cnnPars[0].validNum);
*/


	NVMatrix* cnn1HidVis = new NVMatrix(cnnPars[0].numFilters, \
			cnnPars[0].filterSize * cnnPars[0].filterSize * cnnPars[0].inChannel);
	NVMatrix* cnn1HidBiases = new NVMatrix(cnnPars[0].numFilters, 1);
	NVMatrix* avgOut = new NVMatrix(avgOutLen / logistic->numOut, logistic->numOut);
	NVMatrix* outBiases = new NVMatrix(1, logistic->numOut);

	NVMatrix* cnn2HidVis = new NVMatrix(cnnPars[1].numFilters, \
			cnnPars[1].filterSize * cnnPars[1].filterSize * cnnPars[1].inChannel);
	NVMatrix* cnn2HidBiases = new NVMatrix(cnnPars[1].numFilters, 1);

	gaussRand(cnn1HidVis, 0.01);
//	initW(cnn1HidVis);
	initW(cnn2HidVis);
	cudaMemset(cnn1HidBiases->getDevData(), 0, sizeof(float) * cnn1HidBiasLen);
	cudaMemset(cnn2HidBiases->getDevData(), 0, sizeof(float) * cnn2HidBiasLen);
//	initW(avgOut);
	gaussRand(avgOut, 0.1);
//	cudaMemset(avgOut->getDevData(), 0, sizeof(float) * avgOutLen);
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
			* cnnPars->filterSize * cnnPars->inChannel;
	int cnn1HidBiasLen = cnnPars->numFilters;
	int cnn2HidVisLen = cnnPars[1].numFilters * cnnPars[1].filterSize \
			* cnnPars[1].filterSize * cnnPars[1].inChannel;
	int cnn2HidBiasLen = cnnPars[1].numFilters;
	int avgOutLen = cnnPars[1].poolResultSize * cnnPars[1].poolResultSize * cnnPars[1].numFilters * logistic->numOut;
//	int avgOutLen = cnnPars[0].inSize * cnnPars[0].inSize * cnnPars[0].inChannel* logistic->numOut;
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

	NVMatrix* cnn1TrainData = new NVMatrix(cnnPars->trainNum, cnn1InLen);
	NVMatrix* cnn1TrainLabel = new NVMatrix(cnnPars->trainNum, 1);
	NVMatrix* cnn1ValidData = new NVMatrix(cnnPars->validNum, cnn1InLen);
	NVMatrix* cnn1ValidLabel = new NVMatrix(cnnPars->validNum, 1);

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

	NVMatrix* cnn1_y_i;
	NVMatrix* cnn1_dE_dy_i;
	NVMatrix* cnn2_y_i;
	NVMatrix* cnn2_dE_dy_i;

	clock_t t;
	t = clock();

	for(int epochIdx = 0; epochIdx < cnnPars->numEpoches; epochIdx++){
		int error = 0;

if(epochIdx > 10){
//	cnn1HidVis->showValue("whk");
//	cnn1HidBiases->showValue("hidBias");
//	avgOut->showValue("wij");
//	outBiases->showValue("outBias");
}
		for(int batchIdx = 0; batchIdx < cnnPars->numMinibatches; batchIdx++){

			miniData->changePtrFromStart(cnn1TrainData->getDevData(), \
					miniDataLen * batchIdx);
			miniLabel->changePtrFromStart(cnn1TrainLabel->getDevData(), \
					miniLabelLen * batchIdx);
			cnn1.computeConvOutputs(miniData);
			cnn1.computeMaxOutputs();
			cnn1_y_i = cnn1.getYI();
			cnn1_dE_dy_i = cnn1.getDEDYI();

			cnn2.computeConvOutputs(cnn1_y_i);			
			cnn2.computeMaxOutputs();
			cnn2_y_i = cnn2.getYI();
			cnn2_dE_dy_i = cnn2.getDEDYI();

			layer3.computeClassOutputs(cnn2_y_i);
//			layer3.computeClassOutputs(miniData);
			layer3.computeError(miniLabel, error);
//			layer3.computeDerivs(miniData, miniLabel);
			layer3.computeDerivs(cnn2_y_i, miniLabel, cnn2_dE_dy_i);
			cnn2.computeDerivs(cnn1_y_i);
			cnn2.computeDerivsToIn(cnn1_dE_dy_i);
			cnn1.computeDerivs(miniData);		
	
			cnn1.updatePars();
			cnn2.updatePars();
			layer3.updatePars();

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
					cnn1_y_i = cnn1.getYI();
					cnn2.computeConvOutputs(cnn1_y_i);
					cnn2.computeMaxOutputs();
					cnn2_y_i = cnn2.getYI();

					layer3.computeClassOutputs(cnn2_y_i);
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
					cout << "epochIdx: " << epochIdx << ", error: " \
						<<  (float)totalValid/cnnPars->validNum \
						<< ",likelihood: "<< loglihoodValid<< endl;
			}
		}
		if((epochIdx + 1) % 10 == 0){
			cnn1.transfarLowerPars();
			cnn2.transfarLowerPars();
			layer3.transfarLowerPars();
		} 

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
	
	pars* cnnPars = new pars[numCnnLayers];
	pars* logistic = new pars;

	cnnPars[0].epsHidVis = 0.1;
	cnnPars[0].epsHidBias = 0.2;
	cnnPars[0].mom = 0.9;
	cnnPars[0].wcHidVis = 0;
	cnnPars[0].inSize = 28; 
	cnnPars[0].inChannel = 1;
	cnnPars[0].filterSize = 5;
	cnnPars[0].numFilters = 20; 
	cnnPars[0].stepSize = 1;
	cnnPars[0].convResultSize = (cnnPars[0].inSize - cnnPars[0].filterSize) / cnnPars[0].stepSize + 1;
	cnnPars[0].poolSize = 2;
	cnnPars[0].poolResultSize = cnnPars[0].convResultSize / cnnPars[0].poolSize;
	cnnPars[0].trainNum = 50000;
	cnnPars[0].validNum = 10000;
	cnnPars[0].minibatchSize = 100;
	cnnPars[0].numMinibatches = cnnPars[0].trainNum / (cnnPars[0].minibatchSize * (numProcess - 1));
	cnnPars[0].numValidBatches = cnnPars[0].validNum / (cnnPars[0].minibatchSize * (numProcess - 1));
	cnnPars[0].numEpoches = 100; 
	cnnPars[0].nPush =10;
	cnnPars[0].nFetch = 19;
	cnnPars[0].finePars = 0.95;

	cnnPars[1].epsHidVis = 0.1;
	cnnPars[1].epsHidBias = 0.2;
	cnnPars[1].mom = 0.9;
	cnnPars[1].wcHidVis = 0;
	cnnPars[1].inSize = cnnPars[0].poolResultSize; 
	cnnPars[1].inChannel = cnnPars[0].numFilters;
	cnnPars[1].filterSize = 5;
	cnnPars[1].numFilters = 50; 
	cnnPars[1].stepSize = 1;
	cnnPars[1].convResultSize = (cnnPars[1].inSize - cnnPars[1].filterSize) / cnnPars[1].stepSize + 1;
	cnnPars[1].poolSize = 2;
	cnnPars[1].poolResultSize = cnnPars[1].convResultSize / cnnPars[1].poolSize;
	cnnPars[1].trainNum = cnnPars[0].trainNum;
	cnnPars[1].validNum = cnnPars[0].validNum;
	cnnPars[1].minibatchSize = cnnPars[0].minibatchSize;
	cnnPars[1].numMinibatches = cnnPars[1].trainNum / (cnnPars[1].minibatchSize * (numProcess - 1));
	cnnPars[1].numValidBatches = cnnPars[1].validNum / (cnnPars[1].minibatchSize * (numProcess - 1));

	logistic->wcAvgOut = 0;
	logistic->epsAvgOut = 0.1;
	logistic->epsOutBias = 0.2;
	logistic->numIn = cnnPars[1].poolResultSize * cnnPars[1].poolResultSize * cnnPars[1].numFilters;
//	logistic->numIn = cnnPars[0].inSize * cnnPars[0].inSize * cnnPars[0].inChannel;
	logistic->mom = 0.9;
	logistic->numOut = 10; 
	logistic->minibatchSize = cnnPars[0].minibatchSize;
	logistic->finePars = 0.95;

	if(rank == 0){ 
		managerNode(cnnPars, logistic);
	}   
	else{
		workerNode(cnnPars, logistic);
	} 	

	delete[] cnnPars;
	delete logistic;
	MPI_Finalize();
	return 0;
}



















