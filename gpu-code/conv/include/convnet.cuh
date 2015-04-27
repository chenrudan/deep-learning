/*
 * filename: convnet.cuh
 */

#ifndef CONVNET_H_
#define CONVNET_H_

#include <iostream>
#include "layer.hpp"
#include "utils.cuh"
#include "matrix.h"
#include "nvmatrix.cuh"

class ConvNet : public Layer{

private:
	
	NVMatrix* unrolled_x1;
	NVMatrix* unranged_y;
	NVMatrix* unrolled_x2;
	NVMatrix* ranged_dE_dx;
	NVMatrix* dE_db_tmp;
	NVMatrix* unrolled_conv;
	NVMatrix* ranged_w;
	NVMatrix* unranged_in;

	NVMatrix* _dE_dx_sigmoid;
		
	int _filt_pixs;
	int _conv_pixs;

public:
	ConvNet(pars* network);
	~ConvNet();
	
	void initCuda();
	void computeOutputs(NVMatrix* x);
	void computeDerivsOfPars(NVMatrix* x);
	void computeDerivsOfInput(NVMatrix* dE_dx);
//	void updatePars();


};

#endif
