/*************************************************************************
  > File Name: main.cpp
  > Author: chenrudan
  > Mail: chenrudan123@gmail.com 
  > Created Time: 2014年11月21日 星期五 09时26分54秒
 ************************************************************************/

#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cublas.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <time.h>
#include "mlp.cuh"

using namespace std;

int main(){

    int input_units = 44*44;
    int hidden_units = 1024;
    int out_units = 121;
    int minibatch = 20;
    float lr = -0.01;
    int n_epochs = 1000;
    int n_batchs = 25280/minibatch;
    int valid_size = 5056;
    int valid_minibatch = 16;
    int valid_batchs = valid_size / valid_minibatch;
/*
    int input_units = 784;
    int hidden_units = 500;
    int out_units = 10;
    int minibatch = 20;
    float lr = -0.01;
    int n_epochs = 100;
    int n_batchs = 50000/minibatch;
    int valid_size = 10000;
    int valid_minibatch = 20;
    int valid_batchs = valid_size / valid_minibatch;
*/
    unsigned int h_compute_size = 4;
    unsigned int y_compute_size = 1;

    float l2_reg = 0.0001;
    int patience = 100000;
    int patience_inc = 2;
    float best_valid_loss = 10000;
    float this_valid_loss = 0;
    float imporve_threshold = 0.995;
    bool done_loop = false;
    int valid_freq = (n_batchs < patience / 2) ? n_batchs : (patience / 2) ;

    MLP mlp(input_units, hidden_units , out_units, minibatch);

    float *x_train = new float[minibatch*input_units];
    char *r_x_train = new char[minibatch*input_units];
    int *t_train = new int[minibatch];
    char *r_target_train = new char[minibatch];
    //each epoch try valid
    float *x_valid = new float[input_units * valid_minibatch];
    char *r_x_valid = new char[input_units * valid_minibatch];
    int *t_valid = new int[valid_minibatch];
    char *r_target_valid = new char[valid_minibatch];

    /*    //ifstream fin1("yaleface1_outcome_train.dat", ios::binary);
          ifstream fin1("yalefaces_train.bin", ios::binary);
          ifstream fin2("yale_label_train.bin", ios::binary);
    //ifstream fin3("yaleface1_outcome_valid.dat", ios::binary);
    ifstream fin3("yalefaces_valid.bin", ios::binary);
    ifstream fin4("yale_label_valid.bin", ios::binary);
     */
    /*    ifstream fin1("orlface1_outcome_train.dat", ios::binary);
    //ifstream fin1("orlfaces_train.bin", ios::binary);
    ifstream fin2("orl_label_train.bin", ios::binary);
    ifstream fin3("orlface1_outcome_valid.dat", ios::binary);
    //ifstream fin3("orlfaces_valid.bin", ios::binary);
    ifstream fin4("orl_label_valid.bin", ios::binary);
     */
    /*
    ifstream fin1("lfwface1_outcome_train.dat", ios::binary);
    //ifstream fin1("lfwfaces_train.bin", ios::binary);
    ifstream fin2("lfw_label_train.bin", ios::binary);
    ifstream fin3("lfwface1_outcome_valid.dat", ios::binary);
    //ifstream fin3("lfwfaces_valid.bin", ios::binary);
    ifstream fin4("lfw_label_valid.bin", ios::binary);
*/
    //ifstream fin1("ocean1_outcome_train.bin", ios::binary);
    ifstream fin1("ocean_train.bin", ios::binary);
    ifstream fin2("ocean_label_train.bin", ios::binary);
    //ifstream fin3("ocean1_outcome_valid.bin", ios::binary);
    ifstream fin3("ocean_valid.bin", ios::binary);
    ifstream fin4("ocean_label_valid.bin", ios::binary);

    /*    ifstream fin1("mnist_train.bin", ios::binary);
          ifstream fin2("mnist_label_train.bin", ios::binary);
          ifstream fin3("mnist_valid.bin", ios::binary);
          ifstream fin4("mnist_label_valid.bin", ios::binary);
     */
    clock_t t;
    t = clock();
    mlp.initW();


    cout << "======================================\n";
    cout << "n_epochs: " << n_epochs << endl;
    cout << "n_batchs: " << n_batchs << endl;
    cout << "minibatch: " << minibatch << endl;
    cout << "input_units: " << input_units << endl;
    cout << "hidden_units: " << hidden_units << endl;
    cout << "out_units: " << out_units << endl;
    cout << "lr: " << lr << endl;
    cout << "======================================" << endl;

    int *matrix_train = new int[out_units * out_units];
    int *matrix_valid = new int[out_units * out_units];

    float *save_logloss = new float[valid_size * out_units];

    for(int epoch = 0; epoch < n_epochs && done_loop == false; epoch++){

        /*
        for(int i = 0; i < out_units * out_units; i++){
            matrix_valid[i] = 0;
        }
        for(int i = 0; i < out_units * out_units; i++){
            matrix_train[i] = 0;
        }*/
        

        for(int batch_idx = 0; batch_idx < n_batchs; batch_idx++){

            //read input files
            fin1.seekg(batch_idx * minibatch * input_units * sizeof(float), \
                    fin1.beg);
            fin1.read((char *)x_train, minibatch * input_units * sizeof(float));
           /* fin1.seekg(batch_idx * minibatch * input_units, fin1.beg);
            fin1.read(r_x_train, minibatch * input_units);
               for(int i = 0; i < minibatch * input_units; i++){
               unsigned char tmp = r_x_train[i];
               x_train[i] = (int)tmp / 255.0;
               }*/
             
            fin2.seekg(batch_idx * minibatch, fin2.beg);
            fin2.read(r_target_train, minibatch);
            for(int i = 0; i < minibatch; i++){
                unsigned char tmp = r_target_train[i];
                t_train[i] = (int)tmp;
            }

            mlp.normalizeX(x_train, minibatch, input_units);
            mlp.hidLayer(x_train, h_compute_size);
            mlp.logsiticLayer(y_compute_size);
            int error_train = mlp.errors(t_train, matrix_train);
            float likelihood_train = mlp.negLogLikehood(l2_reg);
            
            mlp.updatePars(x_train, t_train, lr, h_compute_size, \
                    y_compute_size, l2_reg);

            cout << "epoch: " << epoch << ", minibatch: " << batch_idx \
                << ",each minibatch has total number: " <<  minibatch \
                << ", error number: " << error_train \
                << ", error rate: " << (float)error_train/minibatch  \
                << ",negitive likelihood: " << likelihood_train/minibatch << "\n";


            int iter = epoch * n_batchs + batch_idx;

            if((iter + 1) % valid_freq == 0){
            
            /*    for(int i = 0; i < out_units * out_units; i++){
                    cout << matrix_train[i] << " ";
                    if((i + 1) % out_units == 0)
                        cout << endl;
                }*/

                cout << "--------valid for epoch "<< epoch << "--------\n";
                mlp.saveW("w1_t1.bin", "w2_t1.bin", "b1_t1.bin", "b2_t2.bin");

                MLP valid(input_units, hidden_units, out_units, valid_minibatch);

                valid.readW("w1_t1.bin", "w2_t1.bin", "b1_t1.bin", "b2_t2.bin");

                int error_valid = 0;
                float logloss = 0;
                //float likelihood_valid = 0;
                for(int valid_idx = 0; valid_idx < valid_batchs; valid_idx++){

                    fin3.seekg(valid_idx * valid_minibatch * input_units * sizeof(float), \
                            fin3.beg);
                    fin3.read((char *)x_valid, valid_minibatch * input_units * sizeof(float));
                    /*fin3.seekg(valid_idx * valid_minibatch * input_units, fin3.beg);
                    fin3.read(r_x_valid, valid_minibatch * input_units);
                    for(int i = 0; i < valid_minibatch * input_units; i++){
                        unsigned char tmp = r_x_valid[i];
                        x_valid[i] = (int)tmp / 255.0;
                    }*/
                    //
                    fin4.seekg(valid_idx * valid_minibatch, fin4.beg);
                    fin4.read(r_target_valid, valid_minibatch);
                    for(int i = 0; i < valid_minibatch; i++){
                        unsigned char tmp = r_target_valid[i];
                        t_valid[i] = (int)tmp;
                    }

                    valid.normalizeX(x_valid, valid_minibatch, input_units);
                    valid.hidLayer(x_valid, h_compute_size);
                    valid.logsiticLayer(y_compute_size);
                    error_valid += valid.errors(t_valid, matrix_valid);
                    //likelihood_valid += valid.negLogLikehood(l2_reg);
                    float *prob_y = valid.getEva(t_valid, logloss, y_compute_size);
                    memcpy(save_logloss + valid_idx * valid_minibatch, prob_y, \
                            valid_minibatch * out_units * sizeof(float));
                }
                this_valid_loss = (float)error_valid/valid_size;
                cout << "valid total number: " << valid_size << ", error number: " \
                    << error_valid << ", error rate: "<< this_valid_loss  \
                    << "logloss: " << logloss << endl;
                /*  for(int i = 0; i < out_units; i++){
                    for(int j = 0; j < out_units; j++ ){
                        cout << matrix_valid[i * out_units + j];
                        if(i == j)
                            cout << "*";
                        cout << " ";
                    }
                    cout << "\n";
                }
                cout << endl;
                */

            /*    if(this_valid_loss < best_valid_loss){
                    if(this_valid_loss < best_valid_loss * imporve_threshold){
                        patience = (patience > iter * patience_inc) ? patience \
                                   : iter * patience_inc;
                    }
                    best_valid_loss = this_valid_loss;
                }*/
                valid.clear();
            }
/*
            if(patience <= iter){
                done_loop = true;
                cout << "done! best loss: " << best_valid_loss << endl ; 
                break;
            }*/
        }
    }
    ofstream fin5("./ndbs/prob.bin", ios::binary);
    fin5.write((char *)save_logloss, valid_size * out_units * sizeof(float));

    t = clock() - t;
    cout << "This train uses " << (float)t/CLOCKS_PER_SEC << " seconds. \n";
    mlp.clear();
    fin1.close();  
    fin2.close();  
    fin3.close();  
    fin4.close();  
    fin5.close();

    delete[] x_train;
    delete[] t_train;
    delete[] r_x_train;
    delete[] r_target_train;
    delete[] x_valid;
    delete[] r_x_valid;
    delete[] t_valid;
    delete[] r_target_valid;
    /* 
       float *w = new float[k*n];
       float *h = new float[m*n];
       for(int i = 0; i < k*n; i++){
       w[i] = 1;
       }
       for(int i = 0; i < m*n; i++){
       h[i] = 1;
       }
       float *d_x, *d_w, *d_h;
       cout << "success!\n";
       cudaMalloc((void **)&d_x, m*k*sizeof(float));
       cout << "success!\n";
       cudaMalloc((void **)&d_w, k*n*sizeof(float));
       cout << "success!\n";
       cudaMalloc((void **)&d_h, m*n*sizeof(float));
       cout << "success!\n";

       cublasHandle_t handle;
       cublasStatus_t stat;
       cudaError_t cudaStat;

       cublasCreate(&handle);
    //    cudaMemcpy(d_x, x, m*k*sizeof(float), cudaMemcpyHostToDevice);
    //    cudaMemcpy(d_w, w, n*k*sizeof(float), cudaMemcpyHostToDevice);
    //    cudaMemcpy(d_h, h, m*n*sizeof(float), cudaMemcpyHostToDevice);
    cublasSetMatrix(m, k, sizeof(*x), x, m, d_x, m);
    cublasSetMatrix(k, n, sizeof(*w), w, k, d_w, k);
    cublasSetMatrix(m, n, sizeof(*h), h, m, d_h, m);
    float alpha = 1;
    float beta = 0;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, \
    k, &alpha, d_x, m, d_w, k, &beta, d_h, m);

    cout << "success!\n";
    for(int i = 0; i < m*n; i++){
    cout << h[i] << "\n";
    }
    cout << "success!\n";
    //    cudaMemcpy(d_h, h, m*n*sizeof(float), cudaMemcpyDeviceToHost);
    cublasGetMatrix(m, n, sizeof(*h), d_h, m, h, m);
    cout << "success!\n";
    cout << "---h---\n";   
    for(int i = 0; i < m*n; i++){
    cout << h[i] << "\n";
    }
    cout << "---w---\n";
    for(int i = 0; i < 10; i++){
    cout << w[i] << "\n";
    }
     */

    return 0;
}
