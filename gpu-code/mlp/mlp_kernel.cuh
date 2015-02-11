/*
 * mlp_kernel.cuh 
 *
 */

#ifndef MLP_KERNEL_H
#define MLP_KERNEL_H

__global__ void addBias(float* y, float *bias, const unsigned int thread_num, \
        const unsigned int compute_size, const int block_up, \
        const int thread_up);


__global__ void sigmoid(float *x, const unsigned int compute_size, \
        const int block_up, const int thread_up);

__global__ void sigmoidDeriv(float *a, float *b, const float alpha, \
        const int block_up, const int thread_up, \
        const unsigned int compute_size);

__global__ void myExp(float *x, const unsigned int compute_size);

__global__ void myTanh(float *x, const unsigned int compute_size, \
        const int block_up, const int thread_up);

__global__ void myTanhDeriv(float *a, float *b, const float alpha, \
                const int block_up, const int thread_up, \
                        const unsigned int compute_size);

__global__ void mySoftmax(float *x, const unsigned int compute_size, \
        const int block_up);

__global__ void addEleInterval(float *x, const unsigned int interval, \
        const unsigned int compute_size, float *sum, \
        const int block_up, const int thread_up);

__global__ void addEle(float *a, float *b, const float alpha, \
        const float beta,  unsigned int compute_size, \
        const int block_up, const int thread_up);


















#endif /*MLP_KERNEL_H*/
