#include <cuda.h>
#include <cublas.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "cublas_v2.h"

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))
#define m 12
#define n 40
#define k 10

using namespace std;

int main(){
    clock_t t;
    t = clock();
   
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
  
    float *w = new float[m * k];
    float *x = new float[k * n];
    float *h = new float[m * n];

    float ind = 0.1;
    cout << "===w===\n";
    for(int j = 0; j < k; j++){
        for(int i = 0; i < m; i++){
            w[IDX2C(i, j, m)] = ind + 0.01*i;
        }
    }
    for(int i = 0; i < 2 * m; i++){
        for(int j = 0; j < k; j++){
            cout << w[IDX2C(i, j, m)] << "   ";
        }
        cout << "\n";
    }
    ind = 1;
    cout << "===x===\n";
    for(int j = 0; j < n; j++){
        for(int i = 0; i < k; i++){
            x[IDX2C(i, j, k)] = ind + 0.01*i;
        }
    }
    for(int i = 0; i < k; i++){
        for(int j = 0; j < n; j++){
            cout << x[IDX2C(i, j, m)] << "   ";
        }
        cout << "\n";
    }
    ind = 1;
    cout << "===h===\n";
    for(int j = 0; j < n; j++){
        for(int i = 0; i < m; i++){
            h[IDX2C(i, j, m)] = ind + 0.01*i;
        }
    }
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            cout << h[IDX2C(i, j, m)] << "   ";
        }
        cout << "\n";
    }

    float *d_w;
    float *d_x;
    float *d_h;
    cudaStat = cudaMalloc((void**)&d_w, m * k * sizeof(*w));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed\n";
        return EXIT_FAILURE;
    }
    cudaStat = cudaMalloc((void**)&d_x, k * n * sizeof(*x));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed\n";
        return EXIT_FAILURE;
    }
    cudaStat = cudaMalloc((void**)&d_h, m * n * sizeof(*h));
    if (cudaStat != cudaSuccess) {
        cout << "device memory allocation failed\n";
        return EXIT_FAILURE;
    }

    stat = cublasCreate(&handle);
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cout << "CUBLAS initialization failed\n";
        return EXIT_FAILURE;
    }

    //copy matrix from host to device
    stat = cublasSetMatrix(m, k, sizeof(*w), w, m, d_w, m);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cout << "data download failed\n";
        cudaFree(d_w);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix(k, n, sizeof(*x), x, k, d_x, k);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cout << "data download failed\n";
        cudaFree(d_x);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix(m, n, sizeof(*h), h, m, d_h, m);    
    if (stat != CUBLAS_STATUS_SUCCESS) {
        cout << "data download failed\n";
        cudaFree(d_h);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    float al = 1.0f;
    float bet = 1.0f;
    
    filterKernel<<<1, 2>>>(handle, d_w, d_x, d_h, al, bet);
    
    


    stat = cublasGetMatrix(m, n, sizeof(*h), d_h, m, h, m);

    cout << "===h===\n";
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            cout << h[IDX2C(i, j, m)] << "   ";
        }
        cout << "\n";
    }

    cudaFree(d_w); 
    cudaFree(d_x); 
    cudaFree(d_h);
    cublasDestroy(handle);
    delete[] w;
    delete[] x;
    delete[] h;

    t = clock() -t;
    cout << "this train uses " << (float)t/CLOCKS_PER_SEC << "seconds" << endl;
	return EXIT_SUCCESS;  
}

__global__ void filterKernel(cublasHandle_t handle, cublasStatus_t stat, \
	float *d_w, float *d_x, float *d_h, &al, &bet){			
	int start = threadIdx.x * m * k;
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, \
            &al, d_w + start, m, d_x, k, &bet, d_h, m);
}































