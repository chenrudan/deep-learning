///
/// \file data.cu
/// 

#include "data.hpp"

using namespace std;

template <typename Dtype>
void Data<Dtype>::copyFromHost(Dtype* data_value_in, const int data_len){
	cudaError_t status = cudaMemcpy(_data_value, data_value_in, \
			sizeof(Dtype) * data_len, cudaMemcpyHostToDevice);
	if (status != cudaSuccess) {
		cout << stderr, "!!!! device access error (write)\n";
		exit( EXIT_FAILURE );
	}  	
}

template <typename Dtype>
void Data<Dtype>::copyFromDevice(Data* data_in){
	cudaError_t status = cudaMemcpy(_data_value, data_in->getDevData(), \
			sizeof(Dtype) * _amount, cudaMemcpyDeviceToDevice);
	if (status != cudaSuccess) {
		cout << stderr, "!!!! device access error (write)\n";
		exit( EXIT_FAILURE );

	}   
}

template <typename Dtype>
void Data<Dtype>::copyToHost(Dtype* data_value_in, const int data_len){
	cudaError_t status = cudaMemcpy(data_value_in, _data_value, \
			sizeof(Dtype) * data_len, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		cout << stderr, "!!!! device access error (write)\n";
		exit( EXIT_FAILURE );
	}  	
}

template <typename Dtype>
void Data<Dtype>::copyToDevice(Data* data_in){
	cudaError_t status = cudaMemcpy(_data_value, data_in->getDevData(), \
			sizeof(Dtype) * _amount, cudaMemcpyDeviceToDevice);
	if (status != cudaSuccess) {
		cout << stderr, "!!!! device access error (write)\n";
		exit( EXIT_FAILURE );

	}   
}

template <typename Dtype>
void Data<Dtype>::zeros(){
	cudaMemset(_data_value, 0, _amount * sizeof(Dtype));
}

