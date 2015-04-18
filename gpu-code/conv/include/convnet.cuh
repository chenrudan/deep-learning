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
	NVMatrix* _trainData, *_trainLabel;
	NVMatrix* _validData, *_validLabel;

	// ---------------------------
    // Temporary storage
	// ---------------------------
	NVMatrix* _y_h; // conv outputs
	NVMatrix* _y_i; // avg outputs
	NVMatrix* _dE_dy_i, *_dE_dy_h;
	NVMatrix* _dE_dx_h;
	NVMatrix* _dE_dw_hk;
	NVMatrix* _dE_db_h;
	NVMatrix* _dE_dx;
	int* _maxPoolPos;	
	
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

    // ===========================
	// Various learning parameters
    // ===========================
	float _epsHidVis, _epsHidBias;
	float _mom, _wcHidVis, _wcAvgOut;
	float _finePars;
	int _numVis, _numFilters, _numAvg;
	int _minibatchSize;
	int _inSize;
	int _inChannel;
	int _stepSize;
	int _filterSize;
	int _convResultSize;
	int _poolResultSize;
	int _poolSize;

	int _filtPixs;
	int _convPixs;
	int _numTrain;
	int _numValid;	
	cublasHandle_t handle;

public:
	ConvNet(pars* netWork);
	~ConvNet();
	
	void initCuda();
	void computeConvOutputs(NVMatrix* miniData);
	void computeAvgOutputs();
	void computeMaxOutputs();
	void computeDerivs(NVMatrix* miniData);
	void computeDerivsToIn(NVMatrix* downLayer_dE_dy_i);
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

	inline void transfarLowerPars(){
		_epsHidVis *= _finePars;	
		_epsHidBias *= _finePars;
	}

	inline NVMatrix* getHidVis(){
		return _hidVis;
	}
	inline NVMatrix* getHidBias(){
		return _hidBiases;
	}

	inline NVMatrix* getTrainData(){
		return _trainData;
	}
	inline NVMatrix* getTrainLabel(){
		return _trainLabel;
	}
	inline NVMatrix* getValidData(){
		return _validData;
	}
	inline NVMatrix* getValidLabel(){
		return _validLabel;
	}






};

#endif
