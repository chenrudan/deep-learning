
#include <iostream>
//#include <random>
#include <fstream>
#include "utils.cuh"
#include "matrix.h"
//#include "nvmatrix.cuh"

using namespace std;

void initW(NVMatrix* nvMat){
	int length = nvMat->getNumRows() * nvMat->getNumCols();
	float* a = new float[length];
    srand((unsigned)time(NULL));
    float bound = sqrt(1.0 / length);
    for(int i = 0; i < length; i++){
        int k = rand() % 200;
        if(k < 100)
            a[i] = (k/100.0)*(-bound);
        else
            a[i] = ((k - 100)/100.0)*bound; 
    }   
	nvMat->copyFromHost(a, length);
	delete a;
}

void gaussRand(NVMatrix* nvMat, float var, float mean){
    int length = nvMat->getNumRows() * nvMat->getNumCols();
    float* a = new float[length];
 // std::default_random_engine generator;
//  std::normal_distribution<float> distribution(mean, var);

    for(int i = 0; i < length; i++){
//        float k = distribution(generator);
		a[i] = gaussGen(var, mean); 
    } 
    nvMat->copyFromHost(a, length);
	delete a;
}

float gaussGen(float var, float mean)
{
    static float V1, V2, S;
    static int phase = 0;
    float X;
    
    if ( phase == 0 ) {
        do {
            float U1 = (float)rand() / RAND_MAX;
            float U2 = (float)rand() / RAND_MAX;
            
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);
        
        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);
        
    phase = 1 - phase;

    return (X * var + mean);
}



void readPars(Matrix* par, string filename){
    ifstream fin1(filename.c_str(), ios::binary);
    int dataLen = par->getNumRows() * par->getNumCols();
    fin1.read((char*)(par->getData()), sizeof(float) * dataLen);
    fin1.close();
}

void savePars(Matrix* par, string filename){
    ofstream fout(filename.c_str(), ios::binary);
    int dataLen = par->getNumRows() * par->getNumCols();
    fout.write((char*)(par->getData()), sizeof(float) * dataLen);
    fout.close();
}

void readData(NVMatrix* nvData, string filename, bool isData, int addZerosInFront){ 
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
	delete data;
	delete readData;
}

