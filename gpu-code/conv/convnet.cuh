/*
 * filename: convnet.cuh
 */

#ifndef CONVNET_H_
#define CONVNET_H_

#include <iostream>


#include "matrix.h"
#include "nvmatrix.cuh"

class ConvNet {

private:
	// ================
	// Host matrics
	// ================
	Matrix* _hHidVis, *_hHidVisInc;
	Matrix* _hHidBiases, *_hHidBiasInc;
	Matrix* _hAvgOut, *_hAvgOutInc;
	Matrix* _hOutBiases, *_hOutBiasInc;

	// ===========================
    // Device matrices
	// ===========================
	NVMatrix* _hidVis, *_hidVisInc;
	NVMatrix* _hidBiases, *_hidBiasInc;
	NVMatrix* _avgOut, *_avgOutInc;
	NVMatrix* _outBiases, *_outBiasInc;

	// ---------------------------
    // Temporary storage
	// ---------------------------
    NVMatrix* _y_h; // conv outputs
	NVMatrix* _y_i; // avg outputs
	NVMatrix* _y_j; // classification
	NVMatrix* _dE_dy_j, *_dE_db_j, *_dE_dw_ij;
	NVMatrix* _dE_dy_i, *_dE_dy_h;
	NVMatrix* _dE_dx_h;
	NVMatrix* _dE_dw_hk;
	NVMatrix* _dE_db_h;
	int* _maxPoolPos;	

    // ===========================
	// Various learning parameters
    // ===========================
	float _epsHidVis, _epsAvgOut, _epsHidBias, _epsOutBias;
	float _mom, _wcHidVis, _wcAvgOut;
	int _numVis, _numFilters, _numAvg, _numOut;
	int _minibatchSize;
	int _inSize;
	int _inChannel;
	int _filterSize;
	int _outSize;
	int _convResultSize;
	int _poolResultSize;
	
	cublasHandle_t handle;

public:
	ConvNet(Matrix* hHidVis, Matrix* hAvgOut, Matrix* hHidBiases, \
			Matrix* hOutBiases, float epsHidVis, float epsAvgOut, float epsHidBias, \
			float epsOutBias, float mom, float wcHidVis, float wcAvgOut, \
			const int minibatchSize, const int inSize, \
			const int filterSize, const int inChannel, \
			const int numFilters);
	~ConvNet();
	
	void initCuda();
	void computeConvOutputs(NVMatrix* miniData);
	void computeAvgOutputs();
	void computeMaxOutputs();
	void computeClassOutputs();
	double computeError(const NVMatrix *miniLables, int& numError);
	void computeDerivs(NVMatrix* miniData, NVMatrix* miniLabels);
	void updatePars();
	void computeLogistic(NVMatrix* miniData, NVMatrix* miniLabels, bool isTrain);

	inline NVMatrix* getYH(){
		return _y_h;
	}
	inline NVMatrix* getYI(){
		return _y_i;
	}
	inline NVMatrix* getYJ(){
		return _y_j;
	}
	inline NVMatrix* getDEDYJ(){
		return _dE_dy_j;
	}
	inline NVMatrix* getDEDXH(){
		return _dE_dx_h;
	}
	inline NVMatrix* getDEDWIJ(){
		return _dE_dw_ij;
	}
	inline NVMatrix* getDEDWHK(){
		return _dE_dw_hk;
	}
	inline NVMatrix* getDEDBJ(){
		return _dE_db_j;
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
		_epsAvgOut = _epsAvgOut * 0.995;                                                                                                                             
		_epsOutBias = _epsOutBias * 0.995;
	}

	inline NVMatrix* getHidVis(){
		return _hidVis;
	}
	inline NVMatrix* getHidBias(){
		return _hidBiases;
	}
	inline NVMatrix* getAvgOut(){
		return _avgOut;
	}
	inline NVMatrix* getOutBias(){
		return _outBiases;
	}














};

#endif
