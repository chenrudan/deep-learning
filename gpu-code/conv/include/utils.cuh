
#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>
//#include <random>
#include <fstream>
#include "matrix.h"
#include "nvmatrix.cuh"

using namespace std;

typedef struct Pars{
    float epsHidVis;
    float epsHidBias;
    float epsAvgOut;
    float epsOutBias;
    float mom;
    float wcHidVis;
    float wcAvgOut;
	float finePars;

    int inSize;
    int inChannel;
    int filterSize;
    int numFilters;
	int numIn;
    int numOut;
	int stepSize;
	int convResultSize;
	int poolResultSize;
	int poolSize;
    int trainNum;
    int validNum;
    int minibatchSize;
    int numMinibatches;
    int numValidBatches;
    int numEpoches; 

	int nPush;
	int nFetch;
    
}pars;

void initW(NVMatrix* nvMat);

void gaussRand(NVMatrix* nvMat, float var = 1, float mean = 0);

float gaussGen(float var, float mean);

void readPars(Matrix* par, string filename);

void savePars(Matrix* par, string filename);

void readData(NVMatrix* nvData, string filename, bool isData, int addZerosInFront = 0);

#endif
