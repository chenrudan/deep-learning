
#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>
//#include <random>
#include <fstream>
#include "matrix.h"
#include "nvmatrix.cuh"

using namespace std;

typedef struct Pars{
    float w_lr;
    float b_lr;
    float momentum;
    float weight_decay;
	float lr_down_scale;

    int in_size;
	int pad;
    int in_channel;
	int out_size;
    int filter_size;
    int filter_channel;
	int num_in;
    int num_out;
	int stride;
	int padded_in_size;
	int pool_size;
    int num_train;
    int num_valid;
    int minibatch_size;
    int num_minibatch;
    int num_validbatch;
    int num_epoch; 

	int n_push;
	int n_fetch;
    
}pars;

void initW(NVMatrix* nvMat);

void gaussRand(NVMatrix* nvMat, float var = 1, float mean = 0);

float gaussGen(float var, float mean);

void readPars(Matrix* par, string filename);

void savePars(Matrix* par, string filename);

void readData(NVMatrix* nvData, string filename, bool isData, int addZerosInFront = 0);

#endif
