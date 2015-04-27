/*
 * filename: logistic.cuh
 */

#ifndef LOGISTIC_H_
#define LOGISTIC_H_

#include <iostream>

#include "utils.cuh"
#include "layer.hpp"
#include "matrix.h"
#include "nvmatrix.cuh"
#include "nvmatrix_kernel.cuh"

class Logistic : public Layer {

public:
	Logistic(pars* network);
	~Logistic();
	
	void initCuda();
	void computeOutputs(NVMatrix* x);
	double computeError(const NVMatrix* labels, int& num_error);
	void computeDerivsOfPars(NVMatrix* x, NVMatrix* labels);
	void computeDerivsOfInput(NVMatrix* dE_dx);
//	void updatePars();
	

};

#endif
