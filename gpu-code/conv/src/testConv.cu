/*
 * filename:main.cu
 */

#include <iostream>
#include <time.h>

#include "matrix.h"
#include "nvmatrix.cuh"
#include "convnet.cuh"
#include "convnet_kernel.cuh"

using namespace std;

int main(){
	
	float epsHidVis = 0.1;
	float epsHidBias = 0.1;
	float epsAvgOut = 0.1;
	float epsOutBias = 0.1;
	float mom = 0.005;
	float wcHidVis = 0.001;
	float wcAvgOut = 0.001;
	
	int inSize = 28;
	int filterSize = 5;
	int numFilters = 16;
	int numOut = 10;
	int inNum = 16;
	int minibatchSize = 16;
	int numMinibatches = inNum / minibatchSize;
	int inChannel = 1;

	float *data = new float[minibatchSize * inSize * inSize];
	float *label = new float[minibatchSize];
	float *weightHI = new float[numFilters * filterSize * filterSize];
	float *biasHI = new float[numFilters];
	float *weightPO = new float[numFilters * POOL_FORWARD_SIZE \
					  * POOL_FORWARD_SIZE * numOut];
	float *biasPO = new float[numOut];
	
	for(int i = 0; i < minibatchSize; i++){
		for(int j = 0; j < inSize; j++){
			for(int k = 0; k < inSize; k++){
				data[i * inSize * inSize + j * inSize + k] = 1;
			}
		}
		label[i] = 1;
	}
	cout << "done1\n";
	for(int i = 0; i < numFilters; i++){
		for(int j = 0; j < filterSize * filterSize; j++){
			weightHI[i *filterSize * filterSize + j] = 1;
		}
		biasHI[i] = 1;
	}
	cout << "done2\n";
	int lenPO = numFilters*POOL_FORWARD_SIZE*POOL_FORWARD_SIZE;;
	for(int i=0; i<lenPO; i++){
		for(int j = 0; j < numOut; j++){
			weightPO[i * numOut + j] = 1;
		//	cout << weightPO[i * numOut + j] << " ";
			biasPO[j] = 1;
		}
		//cout << endl;
	}
	cout << "done3\n";
	
	Matrix* inData = new Matrix(data, minibatchSize, inSize * inSize);
	NVMatrix* nvData = new NVMatrix(*inData, true);
	Matrix* inLabel = new Matrix(label, minibatchSize, 1);
	NVMatrix* nvLabel = new NVMatrix(*inLabel, true);

	Matrix* hHidVis = new Matrix(weightHI, numFilters, filterSize * filterSize);
	Matrix* hHidBiases = new Matrix(biasHI, numFilters, 1);
	Matrix* hAvgout = new Matrix(weightPO, lenPO, numOut);
	Matrix* hOutBiases = new Matrix(biasPO, 1, numOut);


	ConvNet train(hHidVis, hAvgout, hHidBiases, hOutBiases, epsHidVis, epsAvgOut, \
			epsHidBias, epsOutBias, mom, wcHidVis, wcAvgOut, minibatchSize, \
			inSize, filterSize, inChannel, numFilters);
	cout << "done4\n";

	clock_t t;
	t = clock();
	double loglihood = 0;
	int numError = 0;

	for(int i = 0; i < numMinibatches; i++){
		//读取数据
		/*
		 * Forward pass
		 */
		train.initCuda();
		train.computeConvOutputs(nvData);
		train.computeAvgOutputs();
		train.computeClassOutputs();
		loglihood = train.computeError(inLabel, numError);
		train.computeDerivs(nvData, nvLabel);
		train.updatePars();

	cout << "done5\n";
		
	}
	cout << "error rate: " << (float)numError / inNum << endl;
	t = clock() - t;
	cout << "This train uses " << (float)t/CLOCKS_PER_SEC << " seconds. \n";

	return 0;
}
