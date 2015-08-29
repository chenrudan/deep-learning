///
/// \file python_interface.hpp
///
#ifndef PYTHON_INTERFACE_HPP_
#define PYTHON_INTERFACE_HPP_

#include <iostream>
#include "python2.7/Python.h"

template <typename Dtype>
class PythonInterface {

template<typename D>
friend class TrainDetection;

public:
	PythonInterface();
	~PythonInterface();
	
	void callPythonSaveOutput(string filename, Dtype *data, \
		Dtype *coords, int img_size);
	void callPythonGetCutObject(string filename, Dtype *data, \
		int *coords, int img_size);

private:
};

#include "../src/python_interface.cpp"

#endif
