/*
 * filename: logistic.cuh
 */

#ifndef LOGISTIC_H_
#define LOGISTIC_H_

#include <iostream>

#include "utils.h"
#include "matrix.h"
#include "nvmatrix.cuh"
#include "nvmatrix_kernel.cuh"

class Logistic {

private:
	// ================
	// Host matrics
	// ================
	Matrix* _hAvgOut;
	Matrix* _hOutBiases;

	// ===========================
    // Device matrices
	// ===========================
	NVMatrix* _avgOut, *_avgOutInc;
	NVMatrix* _outBiases, *_outBiasInc;

	// ---------------------------
    // Temporary storage
	// ---------------------------
	NVMatrix* _y_j; // classification
	NVMatrix* _dE_dy_j, *_dE_db_j, *_dE_dw_ij;

    // ===========================
	// Various learning parameters
    // ===========================
	float _epsAvgOut,_epsOutBias;
	float _mom, _wcHidVis, _wcAvgOut;
	float _finePars;
	int _numVis, _numFilters, _numAvg, _numOut;
	int _minibatchSize;
	int _inSize;
	int _inChannel;
	
	cublasHandle_t handle;

public:
	Logistic(Matrix* hAvgOut, Matrix* hOutBiases, pars* netWork);
	~Logistic();
	
	void initCuda();
	void computeClassOutputs(NVMatrix* miniData);
	double computeError(const NVMatrix* const miniLables, int& numError);
	void computeDerivs(NVMatrix* miniData, NVMatrix* miniLabels);
	void updatePars();

	inline NVMatrix* getYJ(){
		return _y_j;
	}
	inline NVMatrix* getDEDYJ(){
		return _dE_dy_j;
	}
	inline NVMatrix* getDEDWIJ(){
		return _dE_dw_ij;
	}
	inline NVMatrix* getDEDBJ(){
		return _dE_db_j;
	}
	inline void transfarLowerPars(){
		_epsAvgOut = _epsAvgOut * _finePars;
		_epsOutBias = _epsOutBias * _finePars;
	}
	inline NVMatrix* getAvgOut(){
		return _avgOut;
	}
	inline NVMatrix* getOutBias(){
		return _outBiases;
	}

};

#endif
