/*
 * filename: nvmatrix.cuh
 */

#ifndef NVMATRIX_H_
#define NVMATRIX_H_

#include <iostream>
#include <string>
#include "cublas_v2.h"
#include "matrix.h"
#include "nvmatrix_kernel.cuh"

using namespace std;

class NVMatrix {
private:
	unsigned int _numCols, _numRows;
	unsigned int _numElements;
	float* _devData;
	bool _ownsData;
	static cudaDeviceProp deviceProps;
	

public:	
	enum FUNCTIONS {LOG, EXP, RECIPROCAL, SOFTMAX};
	enum ALLOCPALCE {ALLOC_ON_GPU_MEMORY, ALLOC_ON_UNIFIED_MEMORY};
    NVMatrix(NVMatrix::ALLOCPALCE a = ALLOC_ON_GPU_MEMORY);
	NVMatrix(int numRows, int numCols, \
			NVMatrix::ALLOCPALCE a = ALLOC_ON_GPU_MEMORY);
	NVMatrix(const Matrix* like, bool copy, \
			NVMatrix::ALLOCPALCE a = ALLOC_ON_GPU_MEMORY);
	NVMatrix(const NVMatrix* like, bool copy, \
			NVMatrix::ALLOCPALCE a = ALLOC_ON_GPU_MEMORY);
	NVMatrix(const NVMatrix* like, \
			NVMatrix::ALLOCPALCE = ALLOC_ON_GPU_MEMORY);
	NVMatrix(const Matrix* like, \
			NVMatrix::ALLOCPALCE = ALLOC_ON_GPU_MEMORY);
	NVMatrix(float* devData, int numRows, int numCols);
	~NVMatrix();
	void _init(unsigned int numRows, unsigned int numCols, \
			NVMatrix::ALLOCPALCE a);
	
	inline bool isSameDims(const Matrix* m) const {
		return m->getNumRows() == _numRows && m->getNumCols() == _numCols;
	}
	inline bool isSameDims(const NVMatrix* m) const { 
		return m->getNumRows() == _numRows && m->getNumCols() == _numCols;
	}
	inline float* getCellPtr(int i, int j) const {
		return &_devData[i * _numCols + j];
	}
	inline int getNumRows() const {
		return _numRows;
	}
	inline int getNumCols() const {
		return _numCols;
	}
	inline int getNumEles() const {
		return _numElements;
	}
	inline float* getDevData() const {
		return _devData;
	}
	inline void changePtr(const int add) {
		_devData = _devData + add;
	}
	inline void changePtrFromStart(float* start, const int add) {
		_devData = start + add;
	}
	inline void setPtr(float* start) {
		_devData = start;
	}
	
	void copyFromHost(const Matrix* hostMatrix);
	void copyFromHost(const Matrix* hostMatrix, bool resizeDeviceMatrix);
	void copyFromHost(float* data, const int dataLength);
	void copyToHost(Matrix* hostMatrix) const;
	void copyToHost(float* data, const int dataLength) const; 
	void copyFromDevice(const NVMatrix* devMatrix);
	void copyFromDevice(const NVMatrix* devMatrix, const int startPos);
    
	void zeros();

	bool resize(int numRows, int numCols);                                  
	bool resize(const NVMatrix* like);
	bool resize(const Matrix* like);
	float* slice(float* &out, int rowStart);

	void checkBounds(int startRow, int endRow, int startCol, int endCol) const;
	NVMatrix* getTranspose();
	void getTranspose(NVMatrix* target);

    void rightMult(NVMatrix* b, float scaleAB, NVMatrix* target, \
			cublasHandle_t& handle);
	void subColVector(NVMatrix* vec);
	void addColVector(NVMatrix* vec);
	void addColVector(NVMatrix* vec, float scaleVec, NVMatrix* target);
	void addRowVector(NVMatrix* vec);
	void addRowVector(NVMatrix* vec, float scaleVec, NVMatrix* target);

	void apply(FUNCTIONS f, NVMatrix* target);
	void apply(FUNCTIONS f);

	void compactCol(NVMatrix* ori, const int internal);	
	void addSum(NVMatrix* b, NVMatrix* c, float scaleThis, \
		     float scaleB, float scaleC);
	void add(NVMatrix* b, float scaleThis, float scale);
	//add multi cols to one col
	NVMatrix* sumCol();
	void sumCol(NVMatrix* target);
	void sumRow(NVMatrix* target);

	void maxPosInRow(NVMatrix* maxVec);
	void eltWiseMultByColVector(NVMatrix* vec);
	void eltWiseMultByColVector(NVMatrix* vec, NVMatrix* target);
	void eltWiseDivideByColVector(NVMatrix* vec);	
	void eltWiseDivideByColVector(NVMatrix* vec, NVMatrix* target);

	void subtractFromScalar(float scalar, NVMatrix* target);
	void subtractFromScalar(float scalar);
	void eltWiseMult(NVMatrix* b, NVMatrix* target);
	void eltWiseMult(NVMatrix* b);
	void showValue(string name);
	void reValue(float value);
	void reValue(int value);
};










#endif
