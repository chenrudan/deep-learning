
#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>
#include <fstream>
#include "matrix.hpp"
#include <time.h>

using namespace std;

void printTime(clock_t &t, string s);


void initW(Matrix<float>* nvMat);

void gaussRand(Matrix<float>* nvMat, float var = 1, \
            float mean = 0);

float gaussGen(float var, float mean);

void readData(Matrix<float>* nvData, string filename, \
            bool isData, int addZerosInFront = 0);

#endif
