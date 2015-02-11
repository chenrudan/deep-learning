/*************************************************************************
  > File Name: mlp.h
  > Author: chenrudan
  > Mail: chenrudan123@gmail.com 
  > Created Time: 2014年11月21日 星期五 09时27分52秒
 ************************************************************************/
#ifndef MLP_H_
#define MLP_H_

#include <cuda.h>
#include <cublas.h>
#include <cuda_runtime.h>
#include <iostream>
#include "cublas_v2.h"

using namespace std;

class MLP{
  private:

    float *_h;
    float *_y;
    //_w_x_h and _w_h_y are matrix, _b_x_h and _b_h_y are vector.
    float *_w_x_h;
    float *_w_h_y;
    float *_b_x_h;
    float *_b_h_y;
    int _num_x;
    int _num_h;
    int _num_y;
    int _minibatch;

    void copyFromHostToDevice(cudaError_t cudaStat, float *host, \
            float *device, int m, int k);
    void copyFromDeviceToHost(cudaError_t cudaStat, float *host, \
            float *device, int m, int k);
    void computeMatMulti(float *a, float *b, float *c, int m, int k, int n, \
            cublasOperation_t op_a, cublasOperation_t op_b);
    void computeMatAddVec(float *x, float *bias, int m, int k, \
            unsigned int compute_size);

  public:

    MLP();
    MLP(int num_x, int num_h, int num_y, int minibatch);
    ~MLP();

    void initPars(int num_x, int num_h, int num_y, int minibatch);
    void initW();
    void normalizeX(float *x, int minibatch, int size);
    void readW(string w_file_x, string w_file_h, string b_file_x, string b_file_h);
    void saveW(string w_file_x, string w_file_h, string b_file_x, string b_file_h);
    
    void hidLayer(float *x, unsigned int h_compute_size);
    void logsiticLayer(unsigned int y_compute_size);
    void updatePars(float *x, int *target, float lr, unsigned int h_compute_size, \
            unsigned int y_compute_size, float l2_reg);
    int errors(int *target, int *martix); 
    float negLogLikehood(float l2_reg);
    void clear();
    float* getWXToH();
    float* getWHToY();
    float* getH();
    float* getY();
    float* getEva(int *target, float &logloss, int compute_size);
};

#endif
