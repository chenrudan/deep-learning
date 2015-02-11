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

MLP::MLP(int num_x, int num_h, int num_y, int minibatch){
    initPars(num_x, num_h, num_y, minibatch);
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

void MLP::initPars(int num_x, int num_h, int num_y, int minibatch){
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

    try{
        this->_w_x_h = new float[num_x * num_h];
    }
    catch(bad_alloc & memExp){
        cerr << memExp.what() << endl;
        abort();
    }
    try{
        this->_w_h_y = new float[num_h * num_y];
    }
    catch(bad_alloc & memExp){
        cerr << memExp.what() << endl;
        abort();
    }
    //init weight, h and y
    this->_b_x_h = new float[num_h];
    this->_b_h_y = new float[num_y];
    this->_h = new float[minibatch * num_h];
    this->_y = new float[minibatch * num_y];

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

//            _w_x_h[pos] = j;
//            cout << _w_x_h[pos] << " ";
        }
  //      cout << endl;
      //  _b_x_h[i] = 1;
    }
    /*
       for(int i = 0; i < _num_h; i++){
       for(int j = 0; j < _num_y; j++){
       _w_h_y[i * _num_y + j] = j;
       }
       _b_h_y[i] = 1;
       }*/
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


void MLP::saveW(string w_file_x, string w_file_h, string b_file_x, \
        string b_file_h){
    ofstream fout1, fout2, fout3, fout4;
    try{
        fout1.open(w_file_x.c_str(), ios::binary|ios::out);
        fout2.open(w_file_h.c_str(), ios::binary|ios::out);
        fout3.open(b_file_x.c_str(), ios::binary|ios::out);
        fout4.open(b_file_h.c_str(), ios::binary|ios::out);
        if(!fout1.is_open() || !fout2.is_open() || !fout3.is_open() \
                || !fout4.is_open())
            throw 1;
    }
    catch(int & value){
        cerr << "Exception opening file\n";
        abort();
    }

    fout1.write((char *)_w_x_h, sizeof(float)*_num_x*_num_h);
    fout2.write((char *)_w_h_y, sizeof(float)*_num_y*_num_h);
    fout3.write((char *)_b_x_h, sizeof(float)*_num_h);
    fout4.write((char *)_b_h_y, sizeof(float)*_num_y);

    fout1.close();
    fout2.close();
    fout3.close();
    fout4.close();
}

void MLP::computeMatMulti(float *a, float *b, float *c, int m, int k, int n, \
        cublasOperation_t op_a, cublasOperation_t op_b){

    cublasHandle_t handle;
    cublasStatus_t stat;
    cudaError_t cudaStat;

    float *d_a, *d_b, *d_c;
    cudaStat = cudaMalloc((void **)&d_a, m * k * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed1\n";
        abort();
    }
    cudaStat = cudaMalloc((void **)&d_b, k * n * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed2\n";
        abort();
    }
    cudaStat = cudaMalloc((void **)&d_c, m * n * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed3\n";
        abort();
    }
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed4\n");
        abort();
    }
    copyFromHostToDevice(cudaStat, a, d_a, m, k);
    copyFromHostToDevice(cudaStat, b, d_b, k, n);
    copyFromHostToDevice(cudaStat, c, d_c, m, n);

    int lda, ldb;
    if(op_a == CUBLAS_OP_N)
        lda = m;
    else{
        lda = k;
    }
    if(op_b == CUBLAS_OP_N)
        ldb = k;
    else
        ldb = n;
    float alpha = 1;
    float beta = 0;
    stat = cublasSgemm(handle, op_a, op_b, m, n, \
            k, &alpha, d_a, lda, d_b, ldb, &beta, d_c, m);

    copyFromDeviceToHost(cudaStat, c, d_c, m, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cublasDestroy(handle);
}

void MLP::copyFromHostToDevice(cudaError_t cudaStat, float *host, \
        float *device, int m, int k){
    //copy matrix from host to device, m*k
    cudaStat = cudaMemcpy(device, host, sizeof(float) * m * k, \
            cudaMemcpyHostToDevice);
    if (cudaStat != cudaSuccess) {
        cout << "data download failed\n";
        cudaFree(device);
        abort();
    }
}

void MLP::copyFromDeviceToHost(cudaError_t cudaStat, float *host, \
        float *device, int m, int k){
    //copy matrix from device to host, m*k
    cudaStat = cudaMemcpy(host, device, sizeof(float) * m * k, \
            cudaMemcpyDeviceToHost);
    if (cudaStat != cudaSuccess) {
        cout << "data download failed\n";
        cudaFree(device);
        abort();
    }
}

void MLP::computeMatAddVec(float *x, float *bias, int m, int k, \
        unsigned int compute_size){
    cudaError_t cudaStat;
    float *d_x, *d_b;
    cudaStat = cudaMalloc((void **)&d_x, m * k * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed5\n";
        abort();
    }   
    cudaStat = cudaMalloc((void **)&d_b, k * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed6\n";
        abort();
    }   
    copyFromHostToDevice(cudaStat, x, d_x, m, k);
    copyFromHostToDevice(cudaStat, bias, d_b, 1, k);
    
    dim3 blocks;
    if(m <= 64 && m > 32)
        blocks = dim3(64, 1, 1);
    else if(m <= 32 && m > 16)
        blocks = dim3(32, 1, 1);
    else if(m <= 16 && m > 4)
        blocks = dim3(16, 1, 1);
    else
        blocks = dim3(4, 1, 1);

    unsigned int thread_size = k/compute_size;
    dim3 threads(thread_size);
/*    if(thread_size <= 1024 && thread_size > 256)
        threads = dim3(1024, 1, 1);
    if(thread_size <= 256 && thread_size > 64)
        threads = dim3(256, 1, 1);
    if(thread_size <= 64 && thread_size > 16)
        threads = dim3(64, 1, 1);
    else
        threads = dim3(16, 1, 1);
*/
    addBias<<<blocks, threads>>>(d_x, d_b, \
            thread_size, compute_size, m, k);
    copyFromDeviceToHost(cudaStat, x, d_x, m, k);

    cudaFree(d_x);
    cudaFree(d_b);

}

void MLP::normalizeX(float *x, int minibatch, int size){
    for(int i = 0; i < minibatch; i++){
        int pos = i * size;
        float min = x[pos];
        float sqrt_sum = 0;
        for(int j = 0; j < size; j++){
            if(min < x[pos + j])
                min = x[pos + j];
        }
        for(int j = 0; j < size; j++){
            x[pos + j] = x[pos + j] - min;
        }
        for(int j = 0; j < size; j++){
            sqrt_sum += x[pos + j] * x[pos + j];
        }
        sqrt_sum = sqrt_sum / size;
        for(int j = 0; j < size; j++){
            x[pos + j] = x[pos + j] / sqrt(sqrt_sum);
        }
    }
}

void MLP::hidLayer(float *x, unsigned int compute_size){
    //compute _w_x_h*x, the cublas is col-major
    computeMatMulti(_w_x_h, x, _h, _num_h, _num_x, _minibatch, \
            CUBLAS_OP_N, CUBLAS_OP_N);
    //compute _w_x_h*x + _b_x_h using kernel function
    computeMatAddVec(_h, _b_x_h, _minibatch, _num_h, compute_size);

    cudaError_t cudaStat;
    float *d_h;
    cudaStat = cudaMalloc((void **)&d_h, _minibatch * _num_h * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed7\n";
        abort();
    }   
    copyFromHostToDevice(cudaStat, _h, d_h, _minibatch, _num_h);

    dim3 blocks;
    if(_minibatch <= 64 && _minibatch > 32)
        blocks = dim3(64, 1, 1);
    else if(_minibatch <= 32 && _minibatch > 16)
        blocks = dim3(32, 1, 1);
    else if(_minibatch <= 16 && _minibatch > 4)
        blocks = dim3(16, 1, 1);
    else
        blocks = dim3(4, 1, 1);

    unsigned int thread_size = _num_h/compute_size;
    dim3 threads(thread_size, 1, 1);
/*    if(thread_size <= 1024 && thread_size > 256)
        threads = dim3(1024, 1, 1);
    if(thread_size <= 256 && thread_size > 64)
        threads = dim3(256, 1, 1);
    if(thread_size <= 64 && thread_size > 16)
        threads = dim3(64, 1, 1);
    else
        threads = dim3(16, 1, 1);
*/
    myTanh<<<blocks, threads>>>(d_h, compute_size, _minibatch, thread_size);

    copyFromDeviceToHost(cudaStat, _h, d_h, _minibatch, _num_h);
/*    cout << "----x----\n";
    for(int i = 0; i < _minibatch; i++){
        for(int j = 0; j < _num_x; j++){
            cout << x[i*_num_x + j] << " ";
        }   
        cout << endl;
    } 
    int tmp;
    if(_num_h > 10)
        tmp = 10;
    else
        tmp = _num_h;
    cout << "----h----\n";
    for(int i = 0; i < _minibatch; i++){
        for(int j = 0; j < _num_h; j++){
            cout << _h[i * _num_h + j] << " ";
        }   
        cout << endl;
    } 
    cout << endl;
    cout << "----b1----\n";
        for(int j = 0; j < _num_h; j++){
            cout << _b_x_h[j] << " ";
        }   
        cout << endl;
    cout << "----haftersigmiod----\n";
    for(int i = 0; i < _minibatch; i++){
        for(int j = 0; j < _num_h; j++){
            cout << _h[i * _num_h + j] << " ";
        }   
        cout << endl;
    } 
    cout << endl;
*/
    cudaFree(d_h);
}

void MLP::logsiticLayer(unsigned int compute_size){
    //compute _h*_w_h_y
    computeMatMulti(_w_h_y, _h, _y, _num_y, _num_h, _minibatch, \
            CUBLAS_OP_N, CUBLAS_OP_N);
    //compute _h*_w_h_y + _b_h_y
    computeMatAddVec(_y, _b_h_y, _minibatch, _num_y, compute_size);
  /*  int tmp;
    if(_num_h > 10)
        tmp = 10;
    else
        tmp = _num_h;
    cout << "----w2----\n";
    for(int i = 0; i < tmp; i++){
        for(int j = 0; j < _num_y; j++){
            cout << _w_h_y[i * _num_y + j] << " ";
        }   
        cout << endl;
    } 
    cout << "----y----\n";
    for(int i = 0; i < _minibatch; i++){
        for(int j = 0; j < _num_y; j++){
            cout << _y[i*_num_y + j] << " ";
        }   
        cout << endl;
    }*/ 

    cudaError_t cudaStat;
    float *d_y;
    cudaStat = cudaMalloc((void **)&d_y, _minibatch * _num_y * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed8\n";
        abort();
    }   
    copyFromHostToDevice(cudaStat, _y, d_y, _minibatch, _num_y);

    dim3 blocks;
    if(_minibatch <= 64 && _minibatch > 32)
        blocks = dim3(64, 1, 1);
    else if(_minibatch <= 32 && _minibatch > 16)
        blocks = dim3(32, 1, 1);
    else if(_minibatch <= 16 && _minibatch > 4)
        blocks = dim3(16, 1, 1);
    else
        blocks = dim3(4, 1, 1);

    compute_size = _num_y;
    unsigned int thread_size = _num_y/compute_size;
    dim3 threads(thread_size, 1, 1);

    mySoftmax<<<blocks, threads>>>(d_y, compute_size, _minibatch);
    copyFromDeviceToHost(cudaStat, _y, d_y, _minibatch, _num_y);
/*    cout << "----yaftersoftmax----\n";
    for(int i = 0; i < _minibatch; i++){
        for(int j = 0; j < _num_y; j++){
            cout << _y[i*_num_y + j] << ":" << j << " ";
        }   
        cout << endl;
    }
*/
    cudaFree(d_y);
}

void MLP::updatePars(float *x, int *target, float lr, \
        unsigned int h_compute_size, unsigned int y_compute_size, \
        float l2_reg){
    float *delta_h = new float[_num_h * _minibatch];
    float *delta_b1 = new float[_num_h];
    float *delta_b2 = new float[_num_y];
    cudaMemset(delta_h, 0, _num_h * _minibatch * sizeof(float));
    cudaMemset(delta_b1, 0, _num_h * sizeof(float));
    cudaMemset(delta_b2, 0, _num_y * sizeof(float));

    cudaError_t cudaStat;
    //compute delta_k = y-t, y is the probability, t is the vector same as [0,0,0,1,0,0...]
    for(int i = 0; i < _minibatch; i++){
        int pos = _num_y * i + target[i];
        _y[pos] -= 1;
    }
    //back-propagate to hidden layer, delta_j = h'(a)*w*delta_k
    //delta_j = (1-h_j^2).*w_h_y*(y-t)
    //h(a) is sigmoid, the gradient is h(a)*(1-h(a)), namely _h.*(1-_h)
    //h(a) is tanh, the gradient is (1-h(a).^2)
    computeMatMulti(_w_h_y, _y, delta_h, _num_h, _num_y, _minibatch, \
            CUBLAS_OP_T, CUBLAS_OP_N);
    
    float *d_delta_h, *d_h;
    cudaStat = cudaMalloc((void **)&d_delta_h, _minibatch * _num_h * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed9\n";
        abort();
    }
    cudaStat = cudaMalloc((void **)&d_h, _minibatch * _num_h * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed10\n";
        abort();
    }
    copyFromHostToDevice(cudaStat, delta_h, d_delta_h, 1, _num_h*_minibatch);
    copyFromHostToDevice(cudaStat, _h, d_h, 1, _num_h*_minibatch);

    dim3 blocks;
    int block_size;
    if(_minibatch <= 64 && _minibatch > 32)
        blocks = dim3(64, 1, 1);
    else if(_minibatch <= 32 && _minibatch > 16)
        blocks = dim3(32, 1, 1);
    else if(_minibatch <= 16 && _minibatch > 4)
        blocks = dim3(16, 1, 1);
    else
        blocks = dim3(4, 1, 1);

    unsigned int thread_size = _num_h/h_compute_size;
    dim3 threads(thread_size, 1, 1);
/*    if(thread_size <= 1024 && thread_size > 256)
        threads = dim3(1024, 1, 1);
    if(thread_size <= 256 && thread_size > 64)
        threads = dim3(256, 1, 1);
    if(thread_size <= 64 && thread_size > 16)
        threads = dim3(64, 1, 1);
    else
        threads = dim3(16, 1, 1);
*/
    myTanhDeriv<<<blocks, threads>>>(d_delta_h, d_h, 1, _minibatch, \
            thread_size, h_compute_size);

    copyFromDeviceToHost(cudaStat, delta_h, d_delta_h, _minibatch, _num_h);
//    cout << "----deltah----\n";
    int tmp;
    if(_num_h > 10)
        tmp = 10;
    else
        tmp = _num_h;
/*    for(int i = 0; i < _minibatch; i++){
        for(int j = 0; j < tmp; j++){
            cout << delta_h[i*_num_h + j] << " ";
        }   
        cout << endl;
    } 
*/
    //update bias, b1 means b_x_h, b2 means b_h_y
    float *d_delta_b1, *d_b1;
    cudaStat = cudaMalloc((void **)&d_delta_b1, _num_h * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed11\n";
        abort();
    }  
    cudaStat = cudaMalloc((void **)&d_b1, _num_h * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed12\n";
        abort();
    }   
    //delta_b1 is the mean of each col of delta_h. 
    copyFromHostToDevice(cudaStat, delta_b1, d_delta_b1, 1, _num_h);
    copyFromHostToDevice(cudaStat, delta_h, d_delta_h, 1, _num_h*_minibatch);
    copyFromHostToDevice(cudaStat, _b_x_h, d_b1, 1, _num_h);

    unsigned int compute_size = _minibatch;
    //set block is h_compute_size, thread is _num_h/h_compute_size
    thread_size = _num_h/h_compute_size;
    blocks = dim3(h_compute_size, 1, 1); 
    threads = dim3(thread_size, 1, 1);
/*
    if(thread_size <= 1024 && thread_size > 256)
        threads = dim3(1024, 1, 1);
    if(thread_size <= 256 && thread_size > 64)
        threads = dim3(256, 1, 1);
    if(thread_size <= 64 && thread_size > 16)
        threads = dim3(64, 1, 1);
    else
        threads = dim3(16, 1, 1);
*/
    addEleInterval<<<blocks, threads>>>(d_delta_h, _num_h, h_compute_size, \
            d_delta_b1, 4, thread_size);
    addEle<<<blocks, threads>>>(d_b1, d_delta_b1, 1, lr, h_compute_size, \
            4, thread_size);

    copyFromDeviceToHost(cudaStat, delta_b1, d_delta_b1, 1, _num_h);
    copyFromDeviceToHost(cudaStat, _b_x_h, d_b1, 1, _num_h);
 /*   cout << "----deltab1----\n";
    for(int i = 0; i <tmp; i++){
        cout << delta_b1[i] << " ";
    } 
    cout << endl;
    cout << "----b1----\n";
    for(int i = 0; i < tmp; i++){
        cout << _b_x_h[i] << " ";
    } 
    cout << endl;
*/
    //delat_b2 is the mean of each col of delta_y namely _y.
    float *d_delta_b2, *d_y, *d_b2;
    cudaStat = cudaMalloc((void **)&d_delta_b2, _num_y * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed12\n";
        abort();
    }
    cudaStat = cudaMalloc((void **)&d_y, _minibatch * _num_y * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed13\n";
        abort();
    }  
    cudaStat = cudaMalloc((void **)&d_b2, _num_y * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed14\n";
        abort();
    }  
/*    cout << "----y-t,delta j----\n";
    for(int i = 0; i < _minibatch; i++){
        for(int j = 0; j < _num_y; j++){
            cout << _y[i*_num_y + j] << " ";
        }   
        cout << endl;
    }
    cout << endl;
*/
    copyFromHostToDevice(cudaStat, _y, d_y, 1, _num_y*_minibatch);
    copyFromHostToDevice(cudaStat, delta_b2, d_delta_b2, 1, _num_y);
    copyFromHostToDevice(cudaStat, _b_h_y, d_b2, 1, _num_y);
    //set block is 1, thread is _num_h
    blocks = dim3(1, 1, 1);
    threads = dim3(_num_y, 1, 1);

    addEleInterval<<<blocks, threads>>>(d_y, _num_y, y_compute_size, \
            d_delta_b2, 1, _num_y);
    addEle<<<blocks, threads>>>(d_b2, d_delta_b2, 1, lr, y_compute_size, \
            1, _num_y);

    copyFromDeviceToHost(cudaStat, delta_b2, d_delta_b2, 1, _num_y);
    copyFromDeviceToHost(cudaStat, _b_h_y, d_b2, 1, _num_y);
  /*  cout << "----deltab2----\n";
    for(int i = 0; i < _num_y; i++){
        cout << delta_b2[i] << " ";
    } 
    cout << endl;
    cout << "----b2----\n";
    for(int i = 0; i < _num_y; i++){
        cout << _b_h_y[i] << " ";
    } 
    cout << endl;
*/
    //update w
    float *delta_w1 = new float[_num_x*_num_h];
    computeMatMulti(delta_h, x, delta_w1, _num_h, _minibatch, _num_x, \
            CUBLAS_OP_N, CUBLAS_OP_T);

    float *d_delta_w1, *d_w1;
    cudaStat = cudaMalloc((void **)&d_delta_w1, _num_x * _num_h * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed15\n";
        abort();
    }   
    cudaStat = cudaMalloc((void **)&d_w1, _num_x * _num_h * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed15\n";
        abort();
    }
    copyFromHostToDevice(cudaStat, _w_x_h, d_w1, 1, _num_x * _num_h);
    copyFromHostToDevice(cudaStat, delta_w1, d_delta_w1, 1, _num_x * _num_h);
   
    if(_num_x/16 > 0){ 
        compute_size = _num_h;
        block_size = 16;
        blocks = dim3(block_size, 1, 1);
        thread_size = _num_x/16;
        threads = dim3(thread_size, 1, 1);
    }
    else{
        compute_size = 1;
        block_size = _num_x;
        blocks = dim3(block_size, 1, 1);
        thread_size = _num_h;
        threads = dim3(thread_size, 1, 1);
    }
    addEle<<<blocks, threads>>>(d_w1, d_delta_w1, 1 + 2*l2_reg*lr, lr / _minibatch, \
            compute_size, block_size, thread_size);

    copyFromDeviceToHost(cudaStat, _w_x_h, d_w1, 1, _num_x * _num_h);
    


    float *delta_w2 = new float[_num_h * _num_y];
    computeMatMulti(_y, _h, delta_w2, _num_y, _minibatch, _num_h, \
            CUBLAS_OP_N, CUBLAS_OP_T);

    float *d_delta_w2, *d_w2;
    cudaStat = cudaMalloc((void **)&d_delta_w2, _num_y * _num_h * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed16\n";
        abort();
    }   
    cudaStat = cudaMalloc((void **)&d_w2, _num_y * _num_h * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed17\n";
        abort();
    }
    copyFromHostToDevice(cudaStat, _w_h_y, d_w2, 1, _num_y * _num_h);
    copyFromHostToDevice(cudaStat, delta_w2, d_delta_w2, 1, _num_y * _num_h);
    
    compute_size = _num_y;
    blocks = dim3(h_compute_size, 1, 1);
    threads = dim3(_num_h/h_compute_size, 1, 1);
    addEle<<<blocks, threads>>>(d_w2, d_delta_w2, 1 + 2*l2_reg*lr, lr / _minibatch, \
            compute_size, h_compute_size, _num_h/h_compute_size);

    copyFromDeviceToHost(cudaStat, _w_h_y, d_w2, 1, _num_y * _num_h);
/*    cout << "----deltaw1----\n";
    int tmp1;
    if(_num_x > 10)
        tmp1 = 10;
    else
        tmp1 = _num_x;
    for(int i = 0; i < tmp1; i++){
        for(int j = 0; j < _num_h; j++){
            cout << lr / _minibatch * delta_w1[i*_num_h + j] << " ";
        }   
        cout << endl;
    }
    cout << "----w1----\n";
    for(int i = 0; i < tmp1; i++){
        for(int j = 0; j < _num_h; j++){
            cout << _w_x_h[i*_num_h + j] << " ";
        }   
        cout << endl;
    }
    cout << "----deltaw2----\n";
    for(int i = 0; i < tmp; i++){
        for(int j = 0; j < _num_y; j++){
            cout << lr / _minibatch * delta_w2[i*_num_y + j] << " ";
        }   
        cout << endl;
    }
    cout << "----w2----\n";
    for(int i = 0; i < tmp; i++){
        for(int j = 0; j < _num_y; j++){
            cout << _w_h_y[i*_num_y + j] << " ";
        }   
        cout << endl;
    }
*/
    delete[] delta_h;
    delete[] delta_b1;
    delete[] delta_b2;
    delete[] delta_w1;
    delete[] delta_w2;
    cudaFree(d_h);
    cudaFree(d_y);
    cudaFree(d_delta_h);
    cudaFree(d_b1);
    cudaFree(d_delta_b1);
    cudaFree(d_delta_b2);
    cudaFree(d_b2);
    cudaFree(d_delta_w2);
    cudaFree(d_w2);
    cudaFree(d_delta_w1);
    cudaFree(d_w1);
}

int MLP::errors(int *target, int *matrix){
    int error = 0;
    for(int i = 0; i < _minibatch; i++){
        int label = 0;
        float tmp = _y[i * _num_y];
        for(int j = 1; j < _num_y; j++){
            int pos = i * _num_y + j;
            if(_y[pos] > tmp){
                label = j;
                tmp = _y[pos];
            }
        }
 //       cout << "--"<< label << target[i] << "--";
        matrix[_num_y * label + target[i]]++;
        if(label != target[i])
            error++;
    }
   // cout << endl;
    return error; 
}

float MLP::negLogLikehood(float l2_reg){
    float loss1 = 0;
    float loss2 = 0;
    for(int i = 0; i < _num_y * _minibatch; i++){
        loss1 -= log(_y[i]);
    }
    for(int i = 0; i < _num_y * _num_h; i++){
        loss2 += _w_h_y[i] * _w_h_y[i];
    }
    for(int i = 0; i < _num_h * _num_x; i++){
        loss2 += _w_x_h[i] * _w_x_h[i];
    }
    loss1 /= _minibatch;
    loss1 += loss2 * l2_reg;
    return loss1;
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

float* MLP::getEva(int *target, float &logloss, int compute_size){
    
    cudaError_t cudaStat;
    float *d_y;
    cudaStat = cudaMalloc((void **)&d_y, _minibatch * _num_y * sizeof(float));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed8\n";
        abort();
    }   
    copyFromHostToDevice(cudaStat, _y, d_y, _minibatch, _num_y);

    dim3 blocks(ceil(_minibatch / 16.0) * 16.0, 1);

    compute_size = _num_y;
    unsigned int thread_size = _num_y/compute_size;
    dim3 threads(thread_size, 1, 1);

    mySoftmax<<<blocks, threads>>>(d_y, compute_size, _minibatch);
    copyFromDeviceToHost(cudaStat, _y, d_y, _minibatch, _num_y);
    
    cudaFree(d_y);
    
    for(int i = 0; i < _minibatch; i++){
        int pos = i * _num_y + target[i];
        double result = _y[pos] < 1 - 10e-15 ? _y[pos] : 1 - 10e-15;
        result = result > 10e-15 ? result : 10e-15;
        logloss += _y[pos];
    }
    
    return this->_y;
}














