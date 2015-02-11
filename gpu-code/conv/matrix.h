/*
 * filename: matrix.h
 */

#ifndef MATRIX_H_                                                                                                                                               
#define MATRIX_H_

#include <limits>
#include <assert.h>
#include <stdio.h>

using namespace std;

class Matrix {

private:
	float *_data;
	bool _ownsData;
	int _numRows, _numCols;
	int _numElements;
	int _numDataBytes;

	void _init(float* data, int numRows, int numCols, bool ownsData);
	void _updateDims(int numRows, int numCols);

public:
	enum FUNCTION {
        TANH, RECIPROCAL, SQUARE, EXP, LOG, ZERO, ONE, LOGISTIC1, LOGISTIC2
	};

	Matrix();
	Matrix(int numRows, int numCols);
	Matrix(float *data, int numRows, int numCols);
	~Matrix();

	inline float& getCell(int i, int j) const {
		return _data[i * _numCols + j];
	}
    inline void setCell(int i, int j, float val) {
		(*this)(i, j) = val;
	}
	float& operator()(int i, int j) const {
		return getCell(i, j);
	}
	inline float* getData() const {
		return _data;
	}
	inline bool isView() const {
		return !_ownsData;
	}
	inline int getNumRows() const {
		return _numRows;
	}
	inline int getNumCols() const {
		return _numCols;
	}
	inline int getNumDataBytes() const {
		return _numDataBytes;
	}
	inline int getNumElements() const {
		return _numElements;
	}
	
	Matrix* getTranspose();
	void apply(Matrix::FUNCTION f); 
	void apply(Matrix::FUNCTION f, Matrix *target);
	void myLog(float *data, int length);
	double sum();
	void showValue(string name);
	void reValue(float value);

};



#endif
