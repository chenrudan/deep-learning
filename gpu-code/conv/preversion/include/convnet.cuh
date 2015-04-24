/*
 * filename: convnet.cuh
 */

#ifndef CONVNET_H_
#define CONVNET_H_

#include <iostream>

#include "utils.h"
#include "matrix.h"
#include "nvmatrix.cuh"

class ConvNet {

private:
	// ================
	// Host matrics
	// ================
	Matrix* _hHidVis, *_hHidVisInc;
	Matrix* _hHidBiases, *_hHidBiasInc;

	// ===========================
    // Device matrices
	// ===========================
	NVMatrix* _hidVis, *_hidVisInc;
	NVMatrix* _hidBiases, *_hidBiasInc;

	// ---------------------------
    // Temporary storage
	// ---------------------------
  NVMatrix* _y_h; // conv outputs
	NVMatrix* _y_i; // avg outputs
	NVMatrix* _dE_dy_i, *_dE_dy_h;
	NVMatrix* _dE_dx_h;
	NVMatrix* _dE_dw_hk;
	NVMatrix* _dE_db_h;
	int* _maxPoolPos;	

    // ===========================
	// Various learning parameters
    // ===========================
	float _epsHidVis, _epsHidBias;
	float _mom, _wcHidVis, _wcAvgOut;
	int _numVis, _numFilters, _numAvg;
	int _minibatchSize;
	int _inSize;
	int _inChannel;
	int _stepSize;
	int _filterSize;
	int _convResultSize;
	int _poolResultSize;
	
	cublasHandle_t handle;

public:
	ConvNet(Matrix* hHidVis, Matrix* hHidBiases, pars* netWork);
	~ConvNet();
	
	void initCuda();
	void computeConvOutputs(NVMatrix* miniData);
	void computeAvgOutputs();
	void computeMaxOutputs();
	void computeDerivs(NVMatrix* miniData, NVMatrix* dE_dy_j, NVMatrix* avgOut);
	void updatePars();

	inline NVMatrix* getYH(){
		return _y_h;
	}
	inline NVMatrix* getYI(){
		return _y_i;
	}
	inline NVMatrix* getDEDXH(){
		return _dE_dx_h;
	}
	inline NVMatrix* getDEDWHK(){
		return _dE_dw_hk;
	}
	inline NVMatrix* getDEDBH(){
		return _dE_db_h;
	}
	inline NVMatrix* getDEDYI(){
		return _dE_dy_i;
	}
	inline NVMatrix* getDEDYH(){
		return _dE_dy_h;
	}
	inline int* getMaxPoolPos(){
		return _maxPoolPos;
	}

	inline void transfarLowerAvgOut(){
	}

	inline NVMatrix* getHidVis(){
		return _hidVis;
	}
	inline NVMatrix* getHidBias(){
		return _hidBiases;
	}








};

#endif
