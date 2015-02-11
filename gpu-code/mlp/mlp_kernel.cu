/*
 * mlp_kernel.cu
 *
 */

#ifndef _MLP_KERNEL_CU_
#define _MLP_KERNEL_CU_

#include <cuda_runtime.h>
#include "mlp_kernel.cuh"


__global__ void addBias(float* y, float *bias, const unsigned int thread_num, \
        const unsigned int compute_size, const int block_up, \
        const int thread_up){
    //y represent output of one layer. 
/*    extern __shared__ float s_bias[];
    for(unsigned int i = 0; i < thread_num; i++){
        s_bias[i] = bias[i];
    }
    __syncthreads();
  */  // 
    if(blockIdx.x < block_up && threadIdx.x < thread_up){
        const unsigned int y_pos = (blockIdx.x * blockDim.x + threadIdx.x) \
                                   * compute_size; 
        const unsigned int b_pos = threadIdx.x * compute_size;
        for(unsigned int i = 0; i < compute_size; i++){
            y[y_pos + i] = y[y_pos + i] + bias[b_pos + i];
        }
    }
}

__global__ void sigmoid(float *x, const unsigned int compute_size, \
        const int block_up, const int thread_up){
    if(blockIdx.x < block_up && threadIdx.x < thread_up){
        const unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) \
                                 * compute_size;
        for(unsigned int i = 0; i < compute_size; i++){
            x[idx + i] = 1 / (1 + __expf(-x[idx + i]));
        }
    }
}

__global__ void sigmoidDeriv(float *a, float *b, const float alpha, \
        const int block_up, const int thread_up, \
        const unsigned int compute_size){
    if(blockIdx.x < block_up && threadIdx.x < thread_up){
        const unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x)
            * compute_size;
        for(unsigned int i = 0; i < compute_size; i++){
            a[idx + i] = alpha * b[idx + i] * (1 - b[idx + i]) \
                         * a[idx + i];
        } 
    }
}

__global__ void myExp(float *x, const unsigned int compute_size){
    const unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) \
                             * compute_size;
    for(unsigned int i = 0; i < compute_size; i++){
        x[idx + i] = __expf(-x[idx + i]);
    }
}

__global__ void myTanh(float *x, const unsigned int compute_size, \
        const int block_up, const int thread_up){
    if(blockIdx.x < block_up && threadIdx.x < thread_up){
        const unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) \
                                 * compute_size;
        for(unsigned int i = 0; i < compute_size; i++){
            x[idx + i] = tanhf(x[idx + i]);
        }
    }
}

__global__ void myTanhDeriv(float *a, float *b, const float alpha, \
        const int block_up, const int thread_up, \
        const unsigned int compute_size){
    if(blockIdx.x < block_up && threadIdx.x < thread_up){
        const unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x)
            * compute_size;
        for(unsigned int i = 0; i < compute_size; i++){
            a[idx + i] = alpha * (1 - b[idx + i] * b[idx + i]) \
                         * a[idx + i];
        } 
    }
}


__global__ void mySoftmax(float *x, const unsigned int compute_size, \
        const int block_up){
    if(blockIdx.x < block_up){
        const unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) \
                                 * compute_size;
        float soft_sum = 0;
        for(unsigned int i = 0; i < compute_size; i++){
            x[idx + i] = __expf(x[idx + i]);
            soft_sum += x[idx + i];
        }
        for(unsigned int i = 0; i < compute_size; i++){
            x[idx + i] = x[idx + i] / soft_sum;
        }
    }
}

__global__ void myEva(float *x, const unsigned int compute_size, \
        const int block_up){
    if(blockIdx.x < block_up){
        const unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) \
                                 * compute_size;
        float sum = 0;
        for(unsigned int i = 0; i < compute_size; i++){
            sum += x[idx + i];
        }
        for(unsigned int i = 0; i < compute_size; i++){
            x[idx + i] = x[idx + i] / sum;
        }
    }
}

__global__ void addEleInterval(float *x, const unsigned int interval, \
        const unsigned int compute_size, float *sum, \
        const int block_up, const int thread_up){
    if(blockIdx.x < block_up && threadIdx.x < thread_up){
        const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        sum[idx] = 0;
        for(int i = 0; i < compute_size; i++){
            sum[idx] += x[idx + i*interval];
        }
        sum[idx] = sum[idx] / compute_size;
    }
}

__global__ void addEle(float *a, float *b, const float alpha, \
        const float beta, unsigned int compute_size, \
        const int block_up, const int thread_up){
    if(blockIdx.x < block_up && threadIdx.x < thread_up){
        const unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) \
                                 * compute_size;
        for(unsigned int i = 0; i < compute_size; i++){
        a[idx + i] = alpha * a[idx + i] + beta * b[idx + i];
        }
    }
}














#endif /*_MLP_KERNEL_CU_*/



















