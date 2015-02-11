/*************************************************************************
  > File Name: mlp.cpp
  > Author: chenrudan
  > Mail: chenrudan123@gmail.com 
  > Created Time: 2014年11月21日 星期五 09时27分47秒
 ************************************************************************/

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include "mlp.cuh"
#include "cublas_v2.h"
#include "mlp_kernel.cuh"

using namespace std;

MLP::MLP(){}

MLP::MLP(int num_x, int num_h, int num_y, int minibatch, int num_label){
    initPars(num_x, num_h, num_y, minibatch, num_label);
}

MLP::~MLP(){
}

void MLP::clear(){
    delete[] _w_x_h;
    delete[] _w_h_y;
    delete[] _b_x_h;
    delete[] _b_h_y;
    delete[] _h;
    delete[] _y;
}

void MLP::initPars(int num_x, int num_h, int num_y, int minibatch, int num_label){
    try{
        if(num_x <= 0 || num_h <= 0 || num_y <= 0)
            throw 1;
        else if(sizeof(num_x) != sizeof(int) || sizeof(num_h) != sizeof(int) \
                || sizeof(num_y) != sizeof(int))
            throw 0.1;
    }
    catch(int & value){
        cout << "Number of input units, hidden units or output units \
            cannot be less than zero!" << endl;
        abort();
    }
    catch(float & value){
        cout << "Number of input units, hidden units or output units \
            must be integer!" << endl;
        abort();
    }
    this->_num_x = num_x;
    this->_num_h = num_h;
    this->_num_y = num_y;
    this->_minibatch = minibatch;
    this->_num_label = num_label;

    try{
        this->_w_x_h = new float[_num_x * _num_h];
    }
    catch(bad_alloc & memExp){
        cerr << memExp.what() << endl;
        abort();
    }
    try{
        this->_w_h_y = new float[_num_h*_num_y];
    }
    catch(bad_alloc & memExp){
        cerr << memExp.what() << endl;
        abort();
    }
    //init weight, h and y
    _w_x_h = new float[_num_x * _num_h];
    _w_h_y = new float[_num_y * _num_h];
    _b_x_h = new float[_num_h];
    _b_h_y = new float[_num_y];
    _h = new float[_minibatch * _num_h];
    _y = new float[_minibatch * _num_y];

}

void MLP::initW(){ 
    srand((unsigned)time(NULL));
    cudaMemset(_w_h_y, 0, _num_h * _num_y * sizeof(float));        
    cudaMemset(_w_x_h, 0, _num_x * _num_h * sizeof(float));        
    cudaMemset(_b_h_y, 0, _num_y * sizeof(float));        
    cudaMemset(_b_x_h, 0, _num_h * sizeof(float));        
    //_w_h_y stay in zero, _w_x_h become init.
    float bound = sqrt(6.0 / (_num_x + _num_h));
    for(int i = 0; i < _num_x; i++){
        for(int j = 0; j < _num_h; j++){
            int k = rand() % 200;
            int pos = i * _num_h + j;
            if(k < 100)
                _w_x_h[pos] = (k/100.0)*(-bound);
            else
                _w_x_h[pos] = ((k - 100)/100.0)*bound;
        }
    }
}

void MLP::readW(string w_file_x, string w_file_h, string b_file_x, \
        string b_file_h){
    ifstream fin1, fin2, fin3, fin4;
    try{
        fin1.open(w_file_x.c_str(), ios::binary|ios::in);
        fin2.open(w_file_h.c_str(), ios::binary|ios::in);
        fin3.open(b_file_x.c_str(), ios::binary|ios::in);
        fin4.open(b_file_h.c_str(), ios::binary|ios::in);
        if(!fin1.is_open() || !fin2.is_open() || !fin3.is_open() \
                || !fin4.is_open())
            throw 1;
    }
    catch(int & value){
        cerr << "Exception opening file\n";
        abort();
    }

    fin1.read((char *)_w_x_h, sizeof(float)*_num_x*_num_h);
    fin2.read((char *)_w_h_y, sizeof(float)*_num_y*_num_h);
    fin3.read((char *)_b_x_h, sizeof(float)*_num_h);
    fin4.read((char *)_b_h_y, sizeof(float)*_num_y);

    fin1.close();
    fin2.close();
    fin3.close();
    fin4.close();
}


void MLP::computeMatMulti(float *a, float *b, float *c, int m, int k, int n){
    cublasHandle_t handle;
    cublasStatus_t stat;
    cudaError_t cudaStat;

    float *d_a, *d_b, *d_c;
    cudaStat = cudaMalloc((void **)&d_a, m * k * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed\n";
        abort();
    }
    cudaStat = cudaMalloc((void **)&d_b, k * n * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed\n";
        abort();
    }
    cudaStat = cudaMalloc((void **)&d_c, m * n * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed\n";
        abort();
    }

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        abort();
    }

    copyFromHostToDevice(cudaStat, a, d_a, m, k);
    copyFromHostToDevice(cudaStat, b, d_b, k, n);
    copyFromHostToDevice(cudaStat, c, d_c, m, n);

    float alpha = 1;
    float beta = 0;
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, \
            k, &alpha, d_a, m, d_b, k, &beta, d_c, m);

    copyFromDeviceToHost(cudaStat, c, d_c, m, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);
}

void MLP::copyFromHostToDevice(cudaError_t cudaStat, float *host, \
        float *device, int m, int k){
    //copy matrix from host to device, m*k
    cudaStat = cudaMemcpy(device, host, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    if (cudaStat != cudaSuccess) {
        cout << "data download failed\n";
        cudaFree(device);
        abort();
    }
}

void MLP::copyFromDeviceToHost(cudaError_t cudaStat, float *host, \
        float *device, int m, int k){
    //copy matrix from device to host, m*k
    cudaStat = cudaMemcpy(host, device, sizeof(float) * m * k, cudaMemcpyDeviceToHost);
    if (cudaStat != cudaSuccess) {
        cout << "data download failed\n";
        cudaFree(device);
        abort();
    }
}

void MLP::computeMatAddVec(float *x, float *bias, int m, int k){
    for(int i = 0 ; i < k; i++){
        bias[i] = 1;
    }   
    cudaError_t cudaStat;
    float *d_x, *d_b;
    cudaStat = cudaMalloc((void **)&d_x, m * k * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed\n";
        abort();
    }   
    cudaStat = cudaMalloc((void **)&d_b, k * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed\n";
        abort();
    }   
    copyFromHostToDevice(cudaStat, x, d_x, m, k);
    copyFromHostToDevice(cudaStat, bias, d_b, 1, k);

    unsigned int compute_size = 4;
    unsigned int thread_size = k/compute_size;
    dim3 blocks(_minibatch, 1, 1); 
    dim3 threads(thread_size, 1, 1);
    for(int i = 0; i < 100; i++){
        cout << i << ":" << x[i] << "\t";
    }
    cout << endl;
    addBias<<<blocks, threads, sizeof(float) * _num_h>>>(d_x, d_b, \
            thread_size, compute_size);

    copyFromDeviceToHost(cudaStat, x, d_x, m, k);

    for(int i = 0; i < 100; i++){
        cout << i << ":" << x[i] << "\t";
    }
    cout << endl;
    cudaFree(d_x);
    cudaFree(d_b);

}


void MLP::hidLayer(float *x){
    //compute _w_x_h*x
    computeMatMulti(x, _w_x_h, _h, _minibatch, _num_h, _num_x);
    //compute _w_x_h*x + _b_x_h using kernel function
    computeMatAddVec(_h, _b_x_h, _minibatch, _num_h);
    cudaError_t cudaStat;
    float *d_h;
    cudaStat = cudaMalloc((void **)&d_h, _minibatch * _num_h * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed\n";
        abort();
    }   
    copyFromHostToDevice(cudaStat, _h, d_h, _minibatch, _num_h);

    unsigned int compute_size = 4;
    unsigned int thread_size = _num_h/compute_size;
    dim3 blocks(_minibatch, 1, 1); 
    dim3 threads(thread_size, 1, 1);
    sigmoid<<<blocks, threads>>>(d_h, compute_size);

    copyFromDeviceToHost(cudaStat, _h, d_h, _minibatch, _num_h);

    for(int i = 0; i < 100; i++){
        cout << i << ":" << _h[i] << "\t";
    }
    cout << endl;
    cudaFree(d_h);
}

void MLP::logsiticLayer(){
    //compute _h*_w_h_y
    computeMatMulti(_h, _w_h_y, _y, _minibatch, _num_y, _num_h);
    //


}




















float* MLP::getWXToH(){
    return this->_w_x_h;
}

float* MLP::getWHToY(){
    return this->_w_h_y;
}

float* MLP::getH(){
    return this->_h;
}

float* MLP::getY(){
    return this->_y;
}
















