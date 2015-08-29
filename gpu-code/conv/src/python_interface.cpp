///
/// \file python_interface.cpp
/// @brief
#include "python_interface.hpp"

using namespace std;

template <typename Dtype>
PythonInterface<Dtype>::PythonInterface() {
	Py_Initialize();
}

template <typename Dtype>
PythonInterface<Dtype>::~PythonInterface() {
	Py_Finalize(); 
}

template <typename Dtype>
void PythonInterface<Dtype>::callPythonGetCutObject(string filename, Dtype *data, \
		int *coords, int img_size) {
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('./script/')");
	PyObject *py_file = PyString_FromString("save_object");
	PyObject *py_module = PyImport_Import(py_file);
	if (!py_module)
	{
		cout << "can't find python file!" << endl;
		return;				               
	}
	PyObject *py_func = PyObject_GetAttrString(py_module, "cutTrainObjectForClassify");
	if(!py_func && !PyCallable_Check(py_func)){
		cout << "can't find python function!" << endl;
		return;
	}

	PyObject *py_pixel_list = PyList_New(img_size*img_size);
	PyObject *py_coords_list = PyList_New(4*MAX_OBJECT_NUM);
	for(int i = 0; i < img_size*img_size; i++){
		PyObject *item = Py_BuildValue("f", data[i]);
		PyList_SET_ITEM(py_pixel_list, i, item);
	}
	for(int i = 0; i < 4*MAX_OBJECT_NUM; i++){
		PyObject *item = Py_BuildValue("i", coords[i]);
		PyList_SET_ITEM(py_coords_list, i, item);
	}

	PyObject *item = Py_BuildValue("s", filename.c_str());

	PyObject *py_params = PyTuple_New(3);
	PyTuple_SetItem(py_params, 0, item);
	PyTuple_SetItem(py_params, 1, py_pixel_list);
	PyTuple_SetItem(py_params, 2, py_coords_list);

	PyObject_CallObject(py_func, py_params);
//	Dtype *h_mini_data = new Dtype[classify_mini_data->getNumEles()];
//	h_mini_data = PyF
//	delete[] h_mini_data;

}

template <typename Dtype>
void PythonInterface<Dtype>::callPythonSaveOutput(string filename, Dtype *data, \
		Dtype *coords, int img_size) {
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('./script/')");
	PyObject *py_file = PyString_FromString("save_object");
	PyObject *py_module = PyImport_Import(py_file);
	if (!py_module)
	{
		cout << "can't find python file!" << endl;
		return;				               
	}
	PyObject *py_func = PyObject_GetAttrString(py_module, "saveObjectCoord");
	if(!py_func && !PyCallable_Check(py_func)){
		cout << "can't find python function!" << endl;
		return;
	}

	PyObject *py_pixel_list = PyList_New(img_size*img_size);
	PyObject *py_coords_list = PyList_New(4*MAX_OBJECT_NUM);
	for(int i = 0; i < img_size*img_size; i++){
		PyObject *item = Py_BuildValue("f", data[i]);
		PyList_SET_ITEM(py_pixel_list, i, item);
	}
	for(int i = 0; i < 4*MAX_OBJECT_NUM; i++){
		PyObject *item = Py_BuildValue("f", coords[i]);
		PyList_SET_ITEM(py_coords_list, i, item);
	}

	PyObject *item = Py_BuildValue("s", filename.c_str());

	PyObject *py_params = PyTuple_New(3);
	PyTuple_SetItem(py_params, 0, item);
	PyTuple_SetItem(py_params, 1, py_pixel_list);
	PyTuple_SetItem(py_params, 2, py_coords_list);

	PyObject_CallObject(py_func, py_params);
}
