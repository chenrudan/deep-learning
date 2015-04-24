/*
 * filename: convnet.cuh
 */

#ifndef CONVNET_H_
#define CONVNET_H_

#include <iostream>

#include "utils.cuh"
#include "matrix.h"
#include "nvmatrix.cuh"

class ConvNet {

private:
	
	NVMatrix* unrolledMiniData1;
	NVMatrix* unrangedYH;
	NVMatrix* hidVis_T;
	NVMatrix* hidBias_T;
	NVMatrix* unrolledMiniData2;
	NVMatrix* rangedDEDXH;
	NVMatrix* dE_dw_hk_T;
	NVMatrix* dE_db_h_tmp;
	NVMatrix* unrolledConv;
	NVMatrix* rangedHidVis;
	NVMatrix* unrangedIn;

	int _filtPixs;
	int _convPixs;

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
