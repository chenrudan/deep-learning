
#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>
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

    int inSize;
    int inChannel;
    int filterSize;
    int numFilters;
    int numOut;
    int trainNum;
    int validNum;
    int minibatchSize;
    int numMinibatches;
    int numValidBatches;
    int numEpoches; 

	int nPush;
	int nFetch;
    
}pars;

inline void initW(float* a, int length){
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

inline void readPars(Matrix* par, string filename){
    ifstream fin1(filename.c_str(), ios::binary);
    int dataLen = par->getNumRows() * par->getNumCols();
    fin1.read((char*)(par->getData()), sizeof(float) * dataLen);
    fin1.close();
}

inline void savePars(Matrix* par, string filename){
    ofstream fout(filename.c_str(), ios::binary);
    int dataLen = par->getNumRows() * par->getNumCols();
    fout.write((char*)(par->getData()), sizeof(float) * dataLen);
    fout.close();
}

inline void readData(NVMatrix* nvData, string filename, bool isData, int addZerosInFront = 0){ 
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

#endif
