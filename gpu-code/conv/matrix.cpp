/*
 * filename: matrix.cpp
 */
#include <iostream>
#include <math.h>
#include "matrix.h"

using namespace std;

Matrix::Matrix(){
	_init(NULL, 0, 0, true);
}

Matrix::Matrix(int numRows, int numCols) {
	_init(NULL, numRows, numCols, true);
	this->_data = numRows * numCols > 0 ? new float[this->_numElements] : NULL;
}

Matrix::Matrix(float* data, int numRows, int numCols) {
	_init(data, numRows, numCols, false);
}

Matrix::~Matrix() {
	if(this->_data != NULL && this->_ownsData) {
		delete[] this->_data;
	}
}

void Matrix::_init(float* data, int numRows, int numCols, bool ownsData) {
	_updateDims(numRows, numCols);
	_ownsData = ownsData;
	_data = data;
}

void Matrix::_updateDims(int numRows, int numCols) {
	this->_numRows = numRows;
	this->_numCols = numCols;
	this->_numElements = numRows * numCols;                                 
	this->_numDataBytes = this->_numElements * sizeof(float);
}

Matrix* Matrix::getTranspose() {
	float *data = _data;
	float *transdata = new float[_numElements];
	for(int i = 0; i < _numRows; i++){
		for(int j = 0; j < _numCols; j++){
			transdata[j * _numRows + i] = data[i * _numCols + j];
		}
	}
	Matrix* trans = new Matrix(transdata, _numCols, _numRows);
	return trans;
}

void Matrix::apply(Matrix::FUNCTION f) {
	apply(f, this);
}

void Matrix::apply(Matrix::FUNCTION f, Matrix *target){
	if(f == LOG){
		myLog(target->getData(), target->getNumRows() * target->getNumCols());		
	}
}

void Matrix::myLog(float *data, int length){
	for(int i = 0; i < length; i++){
		double tmp = data[i] < 1 - 10e-15 ? data[i] : 1 - 10e-15;
		tmp = tmp > 10e-15 ? tmp : 10e-15; 
		data[i] = log(tmp);
	}
}

double Matrix::sum(){
	double result = 0;
	for(int i = 0; i < _numElements; i++){
		result += _data[i];
	}
	return result;
}

void Matrix::showValue(string name){
	cout << "-------------"<< name << "--------------" << endl;
	cout << this->getNumRows() << ":" << this->getNumCols() << endl;
	for(int i = 0; i < this->getNumRows(); i++){
		for(int j = 0; j < this->getNumCols(); j++){
			cout << this->getCell(i, j) << " ";                         
			if(j != 0 && j % (this->getNumCols()) == this->getNumCols()  - 1)
				cout << endl;
		}   
	}   

}

void Matrix::reValue(float value){
	for(int i = 0; i < this->getNumRows(); i++){
		for(int j = 0; j < this->getNumCols(); j++){
			this->getCell(i, j) = value;
		}
	}
}









