/*
 * filename:main.cu
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <omp.h>
#include "mpi.h"
#include "nvmatrix.cuh"
#include "convnet.cuh"
#include "pooling_layer.cuh"
#include "inner_product_layer.cuh"
#include "utils.cuh"
#include "logistic.cuh"
#include "load_layer.hpp"
#include "layer_kernel.cuh"

using namespace std;

#define THREAD_END 100000
enum swapInfo{SWAP_CNN1_W_PUSH, SWAP_CNN1_BIAS_PUSH, \
	SWAP_CNN2_W_PUSH, SWAP_CNN2_BIAS_PUSH,	\
	SWAP_CNN3_W_PUSH, SWAP_CNN3_BIAS_PUSH,	\
	SWAP_INNER1_W_PUSH, SWAP_INNER1_BIAS_PUSH, \
	SWAP_SOFTMAX_W_PUSH, SWAP_SOFTMAX_BIAS_PUSH, \
	SWAP_CNN1_W_FETCH, SWAP_CNN1_BIAS_FETCH, \
	SWAP_CNN2_W_FETCH, SWAP_CNN2_BIAS_FETCH, \
	SWAP_CNN3_W_FETCH, SWAP_CNN3_BIAS_FETCH, \
	SWAP_INNER1_W_FETCH, SWAP_INNER1_BIAS_FETCH, \
	SWAP_SOFTMAX_W_FETCH, SWAP_SOFTMAX_BIAS_FETCH};

int num_process;
int rank;

void managerNode(pars* layer_pars){

	cout << "\n===========overall==============" \
		<< "\ntrain: " << layer_pars[0].num_train \
		<< "\nvalid: " << layer_pars[0].num_valid \
		<< "\nbatchSize: " << layer_pars[0].minibatch_size \
		<< "\nn_fetch: " << layer_pars[0].n_fetch \
		<< "\nn_push: " << layer_pars[0].n_push;

	cout << "\n===========cnn1==============" \
		<< "\nin_size: " << layer_pars[0].in_size \
		<< "\nin_channel: " << layer_pars[0].in_channel \
		<< "\nfilter_size: " << layer_pars[0].filter_size \
		<< "\nfilter_channel: " << layer_pars[0].filter_channel \
		<< "\nstride: " << layer_pars[0].stride \
		<< "\nw_lr: " << layer_pars[0].w_lr \
		<< "\nb_lr: " << layer_pars[0].b_lr \
		<< "\nmomentum: " << layer_pars[0].momentum \
		<< "\nweight_decay: " << layer_pars[0].weight_decay \
		<< "\nlr_scale: " << layer_pars[0].lr_down_scale;

	
	cout << "\n===========pool1==============" \
		<< "\nin_size: " << layer_pars[1].in_size \
		<< "\nin_channel: " << layer_pars[1].in_channel \
		<< "\nstride: " << layer_pars[1].stride \
		<< "\npool_size: " << layer_pars[1].pool_size;
	
	cout << "\n===========cnn2==============" \
		<< "\nin_size: " << layer_pars[2].in_size \
		<< "\nin_channel: " << layer_pars[2].in_channel \
		<< "\nfilter_size: " << layer_pars[2].filter_size \
		<< "\nfilter_channel: " << layer_pars[2].filter_channel \
		<< "\nstride: " << layer_pars[2].stride \
		<< "\nw_lr: " << layer_pars[2].w_lr \
		<< "\nb_lr: " << layer_pars[2].b_lr \
		<< "\nmomentum: " << layer_pars[2].momentum \
		<< "\nweight_decay: " << layer_pars[2].weight_decay \
		<< "\nlr_scale: " << layer_pars[2].lr_down_scale;

	cout << "\n===========pool2==============" \
		<< "\nin_size: " << layer_pars[3].in_size \
		<< "\nin_channel: " << layer_pars[3].in_channel \
		<< "\nstride: " << layer_pars[3].stride \
		<< "\npool_size: " << layer_pars[3].pool_size;

	cout << "\n===========cnn3==============" \
		<< "\nin_size: " << layer_pars[4].in_size \
		<< "\nin_channel: " << layer_pars[4].in_channel \
		<< "\nfilter_size: " << layer_pars[4].filter_size \
		<< "\nfilter_channel: " << layer_pars[4].filter_channel \
		<< "\nstride: " << layer_pars[4].stride \
		<< "\nw_lr: " << layer_pars[4].w_lr \
		<< "\nb_lr: " << layer_pars[4].b_lr \
		<< "\nmomentum: " << layer_pars[4].momentum \
		<< "\nweight_decay: " << layer_pars[4].weight_decay \
		<< "\nlr_scale: " << layer_pars[4].lr_down_scale;

	cout << "\n===========pool3==============" \
		<< "\nin_size: " << layer_pars[5].in_size \
		<< "\nin_channel: " << layer_pars[5].in_channel \
		<< "\nstride: " << layer_pars[5].stride \
		<< "\npool_size: " << layer_pars[5].pool_size;


	cout << "\n===========inner_product1==============" \
		<< "\nnum_in: " << layer_pars[6].num_in \
		<< "\nnum_out: " << layer_pars[6].num_out \
		<< "\nw_lr: " << layer_pars[6].w_lr \
		<< "\nb_lr: " << layer_pars[6].b_lr \
		<< "\nmomentum: " << layer_pars[6].momentum \
		<< "\nweight_decay: " << layer_pars[6].weight_decay \
		<< "\nlr_scale: " << layer_pars[6].lr_down_scale;

	cout << "\n===========softmax==============" \
		<< "\nnum_in: " << layer_pars[7].num_in \
		<< "\nnum_out: " << layer_pars[7].num_out \
		<< "\nw_lr: " << layer_pars[7].w_lr \
		<< "\nb_lr: " << layer_pars[7].b_lr \
		<< "\nmomentum: " << layer_pars[7].momentum \
		<< "\nweight_decay: " << layer_pars[7].weight_decay \
		<< "\nlr_scale: " << layer_pars[7].lr_down_scale << endl;


	int cnn1_in_len = layer_pars[0].in_size * layer_pars[0].in_size * layer_pars[0].in_channel;
	int cnn1_w_len = layer_pars[0].filter_channel * layer_pars[0].filter_size \
			* layer_pars[0].filter_size * layer_pars[0].in_channel;
	int cnn1_b_len = layer_pars[0].filter_channel;

	int cnn2_w_len = layer_pars[2].filter_channel * layer_pars[2].filter_size \
			* layer_pars[2].filter_size * layer_pars[2].in_channel;
	int cnn2_b_len = layer_pars[2].filter_channel;

	int cnn3_w_len = layer_pars[4].filter_channel * layer_pars[4].filter_size \
			* layer_pars[4].filter_size * layer_pars[4].in_channel;
	int cnn3_b_len = layer_pars[4].filter_channel;

	int inner1_w_len = layer_pars[6].num_in * layer_pars[6].num_out;
	int inner1_b_len = layer_pars[6].num_out;

	int softmax_w_len = layer_pars[7].num_in * layer_pars[7].num_out;
	int softmax_b_len = layer_pars[7].num_out;

	int train_data_len_part = layer_pars[0].num_train * cnn1_in_len / (num_process - 1);
	int train_label_len_part = layer_pars[0].num_train / (num_process - 1);
	int valid_data_len_part = layer_pars[0].num_valid * cnn1_in_len / (num_process - 1);
	int valid_label_len_part = layer_pars[0].num_valid / (num_process - 1);

cout << "done8\n";
	NVMatrix* train_data = new NVMatrix(layer_pars[0].num_train, cnn1_in_len);
	NVMatrix* valid_data = new NVMatrix(layer_pars[0].num_valid, cnn1_in_len);
	NVMatrix* train_label = new NVMatrix(layer_pars[0].num_train, 1);
	NVMatrix* valid_label = new NVMatrix(layer_pars[0].num_valid, 1);

/*
    readData(train_data, "../data/input/mnist_train.bin", true);
    readData(valid_data, "../data/input/mnist_valid.bin", true);
    readData(train_label, "../data/input/mnist_label_train.bin", false);
    readData(valid_label, "../data/input/mnist_label_valid.bin", false);
*/

cout << "done7\n";

	ImgInfo<float> *cifar10_info = new ImgInfo<float>;
	LoadCifar10<float> cifar10(cifar10_info);
    for(int i = 1; i < 6; i++){
        string s;
        stringstream ss;
        ss << i;
        ss >> s;    
		string filename = "../data/cifar-10-batches-bin/data_batch_"+s+".bin";
        cifar10.loadBinary(filename, cifar10_info->train_pixel_ptr, \
				cifar10_info->train_label_ptr);    
    }   
    cifar10.loadBinary("../data/cifar-10-batches-bin/test_batch.bin", \
            cifar10_info->test_pixel_ptr, cifar10_info->test_label_ptr);

	train_data->copyFromHost(cifar10_info->train_pixel, layer_pars[0].num_train * cnn1_in_len);
	train_label->copyFromHost(cifar10_info->train_label, layer_pars[0].num_train);
	valid_data->copyFromHost(cifar10_info->test_pixel, layer_pars[0].num_valid * cnn1_in_len);
	valid_label->copyFromHost(cifar10_info->test_label, layer_pars[0].num_valid);


cout << "done6\n";
	NVMatrix* cnn1_w = new NVMatrix(layer_pars[0].filter_size * \
			layer_pars[0].filter_size * layer_pars[0].in_channel, \
			layer_pars[0].filter_channel);
	NVMatrix* cnn1_bias = new NVMatrix(1, layer_pars[0].filter_channel);

	NVMatrix* cnn2_w = new NVMatrix(layer_pars[2].filter_size * \
			layer_pars[2].filter_size * layer_pars[2].in_channel, \
			layer_pars[2].filter_channel);
	NVMatrix* cnn2_bias = new NVMatrix(1, layer_pars[2].filter_channel);

	NVMatrix* cnn3_w = new NVMatrix(layer_pars[4].filter_size * \
			layer_pars[4].filter_size * layer_pars[4].in_channel, \
			layer_pars[4].filter_channel);
	NVMatrix* cnn3_bias = new NVMatrix(1, layer_pars[4].filter_channel);

	NVMatrix* inner1_w = new NVMatrix(inner1_w_len / layer_pars[6].num_out, layer_pars[6].num_out);
	NVMatrix* inner1_bias = new NVMatrix(1, layer_pars[6].num_out);


	NVMatrix* softmax_w = new NVMatrix(softmax_w_len / layer_pars[7].num_out, layer_pars[7].num_out);
	NVMatrix* softmax_bias = new NVMatrix(1, layer_pars[7].num_out);

cout << "done5\n";
	gaussRand(cnn1_w, 0.0001);
//	initW(cnn1_w);
	gaussRand(cnn2_w, 0.001);
//	initW(cnn2_w);
	gaussRand(cnn3_w, 0.01);
//	initW(cnn3_w);
	cudaMemset(cnn1_bias->getDevData(), 0, sizeof(float) * cnn1_b_len);
	cudaMemset(cnn2_bias->getDevData(), 0, sizeof(float) * cnn2_b_len);
	cudaMemset(cnn3_bias->getDevData(), 0, sizeof(float) * cnn3_b_len);

	gaussRand(inner1_w, 0.1);
//	initW(inner1_w);
	cudaMemset(inner1_bias->getDevData(), 0, sizeof(float) * inner1_b_len);
	gaussRand(softmax_w, 0.1);
//	initW(softmax_w);
	cudaMemset(softmax_bias->getDevData(), 0, sizeof(float) * softmax_b_len);

	//	readPars(hHidVis, "hHidVis_t1.bin");
	//	readPars(hHidBiases, "hHidBiases_t1.bin");
	//	readPars(hsoftmax_w, "hsoftmax_w_t1.bin");
	//	readPars(hsoftmax_bias, "hsoftmax_bias_t1.bin");

	MPI_Bcast(cnn1_w->getDevData(), cnn1_w_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn1_bias->getDevData(), cnn1_b_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn2_w->getDevData(), cnn2_w_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn2_bias->getDevData(), cnn2_b_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn3_w->getDevData(), cnn3_w_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn3_bias->getDevData(), cnn3_b_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(inner1_w->getDevData(), inner1_w_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(inner1_bias->getDevData(), inner1_b_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(softmax_w->getDevData(), softmax_w_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(softmax_bias->getDevData(), softmax_b_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	for(int i = 1; i < num_process; i++){
		MPI_Send(train_data->getDevData()+(i-1)*train_data_len_part, train_data_len_part, \
				MPI_FLOAT, i, i, MPI_COMM_WORLD);
		MPI_Send(train_label->getDevData()+(i-1)*train_label_len_part, \
				train_label_len_part, MPI_FLOAT, i, i, MPI_COMM_WORLD);
		MPI_Send(valid_data->getDevData()+(i-1)*valid_data_len_part, valid_data_len_part, \
				MPI_FLOAT, i, i, MPI_COMM_WORLD);
		MPI_Send(valid_label->getDevData()+(i-1)*valid_label_len_part, \
				valid_label_len_part, MPI_FLOAT, i, i, MPI_COMM_WORLD);

	}
	delete cifar10_info;
	delete train_data;
	delete train_label;
	delete valid_data;
	delete valid_label;

	//pro进程，每个进程进行的数据交换次数，0123是push，4567是fetch
	//4个数据地址，8个线程来分别实现两种操作
	const int trans_ops = 20;
	const int num_pars_type = 10;
	float* my_pars[num_pars_type] = {cnn1_w->getDevData(), cnn1_bias->getDevData(), \
			cnn2_w->getDevData(), cnn2_bias->getDevData(), \
			cnn3_w->getDevData(), cnn3_bias->getDevData(), \
			inner1_w->getDevData(), inner1_bias->getDevData(), \
			softmax_w->getDevData(), softmax_bias->getDevData()};
	int pars_len[num_pars_type] = {cnn1_w_len, cnn1_b_len, cnn2_w_len, cnn2_b_len, \
				cnn3_w_len, cnn3_b_len, inner1_w_len, inner1_b_len, \
				softmax_w_len, softmax_b_len};

	#pragma omp parallel num_threads(trans_ops * (num_process - 1)) 
	{

		MPI_Status status;
		int par_state = 0;

		int tid = omp_get_thread_num();
		int pid = tid / trans_ops + 1;
		int swap_id = tid % trans_ops;
		int pars_addr = tid % num_pars_type;

		while(par_state != THREAD_END){
			MPI_Recv(&par_state, 1, MPI_INT, pid, \
					swap_id, MPI_COMM_WORLD, &status);

			if(swap_id < num_pars_type){
				MPI_Recv(my_pars[pars_addr], pars_len[pars_addr], MPI_FLOAT, pid, \
						swap_id+ par_state, MPI_COMM_WORLD, &status);
			}else{
				MPI_Send(my_pars[pars_addr], pars_len[pars_addr], MPI_FLOAT, pid, \
						swap_id + par_state, MPI_COMM_WORLD);
			}   
		}
	}

	delete cnn1_w;
	delete cnn1_bias;
	delete cnn2_w;
	delete cnn2_bias;
	delete cnn3_w;
	delete cnn3_bias;
	delete inner1_w;
	delete inner1_bias;
	delete softmax_w;
	delete softmax_bias;
}


void workerNode(pars* layer_pars){
	
	layer_pars[0].num_train /= (num_process - 1);
	int cnn1_in_len = layer_pars[0].in_size * layer_pars[0].in_size * layer_pars[0].in_channel;

	int cnn1_w_len = layer_pars[0].filter_channel * layer_pars[0].filter_size \
			* layer_pars[0].filter_size * layer_pars[0].in_channel;
	int cnn1_b_len = layer_pars[0].filter_channel;

	int cnn2_w_len = layer_pars[2].filter_channel * layer_pars[2].filter_size \
			* layer_pars[2].filter_size * layer_pars[2].in_channel;
	int cnn2_b_len = layer_pars[2].filter_channel;

	int cnn3_w_len = layer_pars[4].filter_channel * layer_pars[4].filter_size \
			* layer_pars[4].filter_size * layer_pars[4].in_channel;
	int cnn3_b_len = layer_pars[4].filter_channel;

	int inner1_w_len = layer_pars[6].num_in * layer_pars[6].num_out;
	int inner1_b_len = layer_pars[6].num_out;

	int softmax_w_len = layer_pars[7].num_in * layer_pars[7].num_out;
	int softmax_b_len = layer_pars[7].num_out;

	int mini_data_len = layer_pars->minibatch_size * cnn1_in_len;
	int mini_label_len = layer_pars->minibatch_size;

	int train_data_len_part = layer_pars->num_train * cnn1_in_len;
	int train_label_len_part = layer_pars->num_train;
	int valid_data_len_part = layer_pars->num_valid * cnn1_in_len;
	int valid_label_len_part = layer_pars->num_valid;

cout << "done4\n";
	ConvNet cnn1(layer_pars);
	cnn1.initCuda();

	PoolingLayer pool1(layer_pars + 1);
	pool1.initCuda();

	ConvNet cnn2(layer_pars + 2);
	cnn2.initCuda();

	PoolingLayer pool2(layer_pars + 3);
	pool2.initCuda();

	ConvNet cnn3(layer_pars + 4);
	cnn3.initCuda();

	PoolingLayer pool3(layer_pars + 5);
	pool3.initCuda();

	InnerProductLayer inner1(layer_pars + 6);
	inner1.initCuda();

	Logistic softmax1(layer_pars + 7);
	softmax1.initCuda();


	NVMatrix* cnn1_w = cnn1.getW();
	NVMatrix* cnn1_bias = cnn1.getBias();
	NVMatrix* cnn2_w = cnn2.getW();
	NVMatrix* cnn2_bias = cnn2.getBias();
	NVMatrix* cnn3_w = cnn3.getW();
	NVMatrix* cnn3_bias = cnn3.getBias();
	NVMatrix* inner1_w = inner1.getW();
	NVMatrix* inner1_bias = inner1.getBias();
	NVMatrix* softmax_w = softmax1.getW();
	NVMatrix* softmax_bias = softmax1.getBias();

	MPI_Bcast(cnn1_w->getDevData(), cnn1_w_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn1_bias->getDevData(), cnn1_b_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn2_w->getDevData(), cnn2_w_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn2_bias->getDevData(), cnn2_b_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn3_w->getDevData(), cnn3_w_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn3_bias->getDevData(), cnn3_b_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(inner1_w->getDevData(), inner1_w_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(inner1_bias->getDevData(), inner1_b_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(softmax_w->getDevData(), softmax_w_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(softmax_bias->getDevData(), softmax_b_len, MPI_FLOAT, 0, MPI_COMM_WORLD);

	NVMatrix* train_data = new NVMatrix(layer_pars->num_train, cnn1_in_len);
	NVMatrix* train_label = new NVMatrix(layer_pars->num_train, 1);
	NVMatrix* valid_data = new NVMatrix(layer_pars->num_valid, cnn1_in_len);
	NVMatrix* valid_label = new NVMatrix(layer_pars->num_valid, 1);

	NVMatrix* mini_data = new NVMatrix(layer_pars->minibatch_size, cnn1_in_len);
	NVMatrix* mini_label = new NVMatrix(layer_pars->minibatch_size, 1);

	MPI_Status status;
	MPI_Recv(train_data->getDevData(), train_data_len_part, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);
	MPI_Recv(train_label->getDevData(), train_label_len_part, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);
	MPI_Recv(valid_data->getDevData(), valid_data_len_part, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);
	MPI_Recv(valid_label->getDevData(), valid_label_len_part, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);

cout << "done9\n";
	int passMsg = 0;

	NVMatrix* cnn1_y;
	NVMatrix* cnn1_dE_dy;
	NVMatrix* pool1_y;
	NVMatrix* pool1_dE_dy;
	NVMatrix* cnn2_y;
	NVMatrix* cnn2_dE_dy;
	NVMatrix* pool2_y;
	NVMatrix* pool2_dE_dy;
	NVMatrix* cnn3_y;
	NVMatrix* cnn3_dE_dy;
	NVMatrix* pool3_y;
	NVMatrix* pool3_dE_dy;
	NVMatrix* inner1_y;
	NVMatrix* inner1_dE_dy;

	const int num_pars_type = 10;
	float* my_pars[num_pars_type] = {cnn1_w->getDevData(), cnn1_bias->getDevData(), \
			cnn2_w->getDevData(), cnn2_bias->getDevData(), \
			cnn3_w->getDevData(), cnn3_bias->getDevData(), \
			inner1_w->getDevData(), inner1_bias->getDevData(), \
			softmax_w->getDevData(), softmax_bias->getDevData()};
	int pars_len[num_pars_type] = {cnn1_w_len, cnn1_b_len, cnn2_w_len, cnn2_b_len, \
				cnn3_w_len, cnn3_b_len, inner1_w_len, inner1_b_len, \
			     softmax_w_len, softmax_b_len};

	clock_t t;
	t = clock();
	clock_t t1;
	t1 = clock();
	for(int epoch_idx = 0; epoch_idx < layer_pars->num_epoch; epoch_idx++){
		int error = 0;
/*
if(epoch_idx > 1){
	cnn1_w->showValue("cnn1w");
	cnn1_bias->showValue("cnn1b");
	cnn2_w->showValue("cnn2w");
	cnn2_bias->showValue("cnn2b");
	inner1_w->showValue("innerw");
	cnn3_w->showValue("cnn3w");
	cnn3_bias->showValue("cnn3b");
	inner1_w->showValue("inner1w");
	inner1_bias->showValue("inner1b");
	softmax_w->showValue("softmaxw");
	softmax_bias->showValue("softmaxb");
}
*/
		for(int batch_idx = 0; batch_idx < layer_pars->num_minibatch; batch_idx++){

			mini_data->changePtrFromStart(train_data->getDevData(), \
					mini_data_len * batch_idx);
			mini_label->changePtrFromStart(train_label->getDevData(), \
					mini_label_len * batch_idx);
//cout << "\n===================";
//		printTime(t, "start");
			cnn1.computeOutputs(mini_data);
			cnn1_y = cnn1.getY();
//		printTime(t, "cnn1 out");
			pool1.computeOutputs(cnn1_y);
			pool1_y = pool1.getY();
//		printTime(t, "pool out");
			cnn2.computeOutputs(pool1_y);			
			cnn2_y = cnn2.getY();
//		printTime(t, "cnn2 out");
			pool2.computeOutputs(cnn2_y);
			pool2_y = pool2.getY();
//		printTime(t, "pool2 out");
			cnn3.computeOutputs(pool2_y);
			cnn3_y = cnn3.getY();
//		printTime(t, "cnn3 out");
			pool3.computeOutputs(cnn3_y);
			pool3_y = pool3.getY();
//		printTime(t, "pool3 out");
			inner1.computeOutputs(pool3_y);
			inner1_y = inner1.getY();
//		printTime(t, "inner1 out");
			softmax1.computeOutputs(inner1_y);
			softmax1.computeError(mini_label, error);
//		printTime(t, "softmax out");

			softmax1.computeDerivsOfPars(inner1_y, mini_label);
//		printTime(t, "softmax pars");
			inner1_dE_dy = inner1.getDEDY();
			softmax1.computeDerivsOfInput(inner1_dE_dy);
//		printTime(t, "softmax input");
			inner1.computeDerivsOfPars(pool3_y);
//		printTime(t, "inner pars");
			pool3_dE_dy = pool3.getDEDY();
			inner1.computeDerivsOfInput(pool3_dE_dy);
//		printTime(t, "inner input");
			cnn3_dE_dy = cnn3.getDEDY();
			pool3.computeDerivsOfInput(cnn3_dE_dy);
//		printTime(t, "pool3 input");
			cnn3.computeDerivsOfPars(pool2_y);
//		printTime(t, "cnn3 pars");
			pool2_dE_dy = pool2.getDEDY();
			cnn3.computeDerivsOfInput(pool2_dE_dy);
//		printTime(t, "cnn3 input");
			cnn2_dE_dy = cnn2.getDEDY();
			pool2.computeDerivsOfInput(cnn2_dE_dy);
//		printTime(t, "pool2 input");
			cnn2.computeDerivsOfPars(pool1_y);
//		printTime(t, "cnn2 pars");
			pool1_dE_dy = pool1.getDEDY();
			cnn2.computeDerivsOfInput(pool1_dE_dy);
//		printTime(t, "cnn2 input");
			cnn1_dE_dy = cnn1.getDEDY();
			pool1.computeDerivsOfInput(cnn1_dE_dy);
//		printTime(t, "pool1 input");
			cnn1.computeDerivsOfPars(mini_data);
//		printTime(t, "cnn1 pars");

			cnn1.updatePars();
			cnn2.updatePars();
			cnn3.updatePars();
			inner1.updatePars();
			softmax1.updatePars();

			if((batch_idx + 1) % layer_pars->n_push == 0){
				if(epoch_idx == layer_pars->num_epoch - 1){
					if((batch_idx + layer_pars->n_push) >= layer_pars->num_minibatch \
							|| batch_idx == layer_pars->num_minibatch - 1)
						passMsg = THREAD_END;
					else
						passMsg = batch_idx;
				}
				else
					passMsg = batch_idx;
				#pragma omp parallel num_threads(num_pars_type)
				{
					int tid = omp_get_thread_num();
					int pars_addr = tid % num_pars_type;
					int swap_id = tid % num_pars_type;

					MPI_Send(&passMsg, 1, MPI_INT, 0, \
						swap_id, MPI_COMM_WORLD);
					MPI_Send(my_pars[pars_addr], pars_len[pars_addr], \
						MPI_FLOAT, 0, swap_id + passMsg, MPI_COMM_WORLD);
					
				}
			}
			if((batch_idx + 1) % layer_pars->n_fetch == 0){
				if(epoch_idx == layer_pars->num_epoch - 1){
					if((batch_idx + layer_pars->n_fetch) >= layer_pars->num_minibatch \
							|| batch_idx == layer_pars->num_minibatch - 1)
						passMsg = THREAD_END;
					else
						passMsg = batch_idx;
				}else
					passMsg = batch_idx;
			
				#pragma omp parallel num_threads(num_pars_type)
				{
					int tid = omp_get_thread_num();
					int pars_addr = tid % num_pars_type;
					int swap_id = tid % num_pars_type + num_pars_type;

					MPI_Send(&passMsg, 1, MPI_INT, 0, \
						swap_id, MPI_COMM_WORLD);
					MPI_Recv(my_pars[pars_addr], pars_len[pars_addr], \
						MPI_FLOAT, 0, swap_id + passMsg, MPI_COMM_WORLD, &status);
					
				}
			}

			if(batch_idx == layer_pars->num_minibatch - 1){ 
				int errorValid = 0;
				float loglihoodValid = 0;
				for(int validIdx = 0; validIdx < layer_pars->num_validbatch; validIdx++){

					mini_data->changePtrFromStart(valid_data->getDevData(), \
							mini_data_len * validIdx);
					mini_label->changePtrFromStart(valid_label->getDevData(), \
							mini_label_len * validIdx);
					cnn1.computeOutputs(mini_data);
					cnn1_y = cnn1.getY();
					pool1.computeOutputs(cnn1_y);
					pool1_y = pool1.getY();
					cnn2.computeOutputs(pool1_y);			
					cnn2_y = cnn2.getY();
					pool2.computeOutputs(cnn2_y);
					pool2_y = pool2.getY();
					cnn3.computeOutputs(pool2_y);			
					cnn3_y = cnn3.getY();
					pool3.computeOutputs(cnn3_y);
					pool3_y = pool3.getY();
					inner1.computeOutputs(pool3_y);
					inner1_y = inner1.getY();
					softmax1.computeOutputs(inner1_y);
					loglihoodValid += softmax1.computeError(mini_label, errorValid);

				}
				int totalValid = errorValid;
				if(num_process > 2){
					if(rank == 1){
						for(int i = 2; i < num_process; i++){
							MPI_Recv(&errorValid, 1, MPI_INT, i, i, \
									MPI_COMM_WORLD, &status);   
							totalValid += errorValid;
						}       
					}else{  
						MPI_Send(&errorValid, 1, MPI_INT, 1, rank, MPI_COMM_WORLD);
					}       
				}       
				if(rank == 1)
					cout << "epoch_idx: " << epoch_idx << ", error: " \
						<<  (float)totalValid/layer_pars->num_valid \
						<< ",likelihood: "<< loglihoodValid<< endl;
			}
		}
		
		if(rank == 1){
			t1 = clock() - t1;
			cout << " " << ((float)t1/CLOCKS_PER_SEC) << " seconds.\n";
			t1 = clock();
		}
		
		if((epoch_idx + 1) % 5 == 0){
			cout << "-----cnn1-----\n";
			cnn1.transfarLowerPars();
			cout << "-----cnn2-----\n";
			cnn2.transfarLowerPars();
			cout << "-----cnn3-----\n";
			cnn3.transfarLowerPars();
			cout << "-----inner1-----\n";
			inner1.transfarLowerPars();
			cout << "-----softmax1-----\n";
			softmax1.transfarLowerPars();
		}
		if((epoch_idx + 1)% 100 == 0){
        	string s;
        	stringstream ss;
        	ss << epoch_idx;
        	ss >> s;    
			savePars(cnn1_w, "../snapshot/w_snap/cnn1_w_" + s + "_t1.bin");
			cout << s << endl;
			savePars(cnn1_bias, "../snapshot/w_snap/cnn1_bias_" + s + "_t1.bin");
			savePars(cnn2_w, "../snapshot/w_snap/cnn2_w_" + s + "_t1.bin");
			savePars(cnn2_bias, "../snapshot/w_snap/cnn2_bias_" + s + "_t1.bin");
			savePars(cnn3_w, "../snapshot/w_snap/cnn3_w_" + s + "_t1.bin");
			savePars(cnn3_bias, "../snapshot/w_snap/cnn3_bias_" + s + "_t1.bin");
			savePars(inner1_w, "../snapshot/w_snap/inner1_w_" + s + "_t1.bin");
			savePars(inner1_bias, "../snapshot/w_snap/inner1_bias_" + s + "_t1.bin");
			savePars(softmax_w, "../snapshot/w_snap/softmax1_w_" + s + "_t1.bin");
			savePars(softmax_bias, "../snapshot/w_snap/softmax1_bias_" + s + "_t1.bin");
		}


	}
	if(rank == 1){
		t = clock() - t;
		cout << " " << ((float)t/CLOCKS_PER_SEC) / layer_pars->num_epoch << " seconds.\n";
		t = clock();
	}

	delete mini_data;
	delete mini_label;
	delete train_data;
	delete train_label;
	delete valid_data;
	delete valid_label;
}

int main(int argc, char** argv){

	int prov;
	MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE, &prov);
	if (prov < MPI_THREAD_MULTIPLE)
	{   
		printf("Error: the MPI library doesn't provide the required thread level\n");
		MPI_Abort(MPI_COMM_WORLD, 0); 
	}   
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&num_process);

	if(num_process <= 1){
		printf("Error: process number must bigger than 1\n");
		MPI_Abort(MPI_COMM_WORLD, 0); 
	}
	//检测有几个gpu
	int numGpus;
	cudaGetDeviceCount(&numGpus);
	cudaSetDevice(rank%numGpus);

/*
	// Ensure that RDMA ENABLED CUDA is set correctly
    int direct = getenv("MPICH_RDMA_ENABLED_CUDA")==NULL?0:atoi(getenv ("MPICH_RDMA_ENABLED_CUDA"));
    if(direct != 1){
        printf ("MPICH_RDMA_ENABLED_CUDA not enabled!\n");
        exit (EXIT_FAILURE);
    }
*/

	const int num_layer = 8;

	pars* layer_pars = new pars[num_layer];

	layer_pars[0].w_lr = 0.1;
	layer_pars[0].b_lr = 0.1;
	layer_pars[2].w_lr = 0.1;
	layer_pars[2].b_lr = 0.2;
	layer_pars[4].w_lr = 0.01;
	layer_pars[4].b_lr = 0.01;

	layer_pars[6].w_lr = 0.001;
	layer_pars[6].b_lr = 0.005;
	layer_pars[7].w_lr = 0.0005;
	layer_pars[7].b_lr = 0.0005;

	layer_pars[0].lr_down_scale = 0.95;
	layer_pars[2].lr_down_scale = 0.95;
	layer_pars[4].lr_down_scale = 0.95;
	layer_pars[6].lr_down_scale = 0.95;
	layer_pars[7].lr_down_scale = 0.95;

	layer_pars[0].weight_decay = 0;
	layer_pars[2].weight_decay = 0;
	layer_pars[4].weight_decay = 0;
	layer_pars[6].weight_decay = 0;
	layer_pars[7].weight_decay = 0;

//	layer_pars[0].w_lr = 5;
//	layer_pars[0].b_lr = 10;
	layer_pars[0].momentum = 0.9;
	layer_pars[0].in_size = 32; 
	layer_pars[0].in_channel = 3;
	layer_pars[0].filter_size = 5;
	layer_pars[0].filter_channel = 32; 
	layer_pars[0].stride = 1;
	layer_pars[0].pad = 2;
	layer_pars[0].padded_in_size = layer_pars[0].in_size + 2 * layer_pars[0].pad;
	layer_pars[0].out_size = (layer_pars[0].padded_in_size - layer_pars[0].filter_size) / layer_pars[0].stride + 1;
	layer_pars[0].num_train = 50000;
	layer_pars[0].num_valid = 10000;
	layer_pars[0].minibatch_size = 100;
	layer_pars[0].num_minibatch = layer_pars[0].num_train / (layer_pars[0].minibatch_size * (num_process - 1));
	layer_pars[0].num_validbatch = layer_pars[0].num_valid / (layer_pars[0].minibatch_size * (num_process - 1));
	layer_pars[0].num_epoch = 1; 
	layer_pars[0].n_push = 49;
	layer_pars[0].n_fetch = 50;
//	layer_pars[0].lr_down_scale = 0.95;

	layer_pars[1].in_size = layer_pars[0].out_size; 
	layer_pars[1].in_channel = layer_pars[0].filter_channel;
	layer_pars[1].filter_channel = layer_pars[0].filter_channel;
	layer_pars[1].pool_size = 3;
	layer_pars[1].stride = 2;
	layer_pars[1].out_size = ceil(((layer_pars[0].out_size - layer_pars[1].pool_size) * 1.0f) \
					 / layer_pars[1].stride) + 1;
	layer_pars[1].minibatch_size = layer_pars[0].minibatch_size;

//	layer_pars[2].w_lr = 1;
//	layer_pars[2].b_lr = 2;
	layer_pars[2].momentum = 0.9;
	layer_pars[2].in_size = layer_pars[1].out_size; 
	layer_pars[2].in_channel = layer_pars[1].filter_channel;
	layer_pars[2].filter_size = 5;
	layer_pars[2].filter_channel = 32; 
	layer_pars[2].stride = 1;
	layer_pars[2].pad = 2;
	layer_pars[2].padded_in_size = layer_pars[2].in_size + 2 * layer_pars[2].pad;
	layer_pars[2].out_size = (layer_pars[2].padded_in_size - layer_pars[2].filter_size) / layer_pars[2].stride + 1;
	layer_pars[2].minibatch_size = layer_pars[0].minibatch_size;
//	layer_pars[2].lr_down_scale = 0.95;


	layer_pars[3].in_size = layer_pars[2].out_size; 
	layer_pars[3].in_channel = layer_pars[2].filter_channel;
	layer_pars[3].filter_channel = layer_pars[2].filter_channel;
	layer_pars[3].pool_size = 3;
	layer_pars[3].stride = 2;
	layer_pars[3].out_size = ceil(((layer_pars[2].out_size - layer_pars[3].pool_size) * 1.0f)\
					 / layer_pars[3].stride) + 1;
	layer_pars[3].minibatch_size = layer_pars[2].minibatch_size;

//	layer_pars[4].w_lr = 1;
//	layer_pars[4].b_lr = 2;
	layer_pars[4].momentum = 0.9;
	layer_pars[4].in_size = layer_pars[3].out_size; 
	layer_pars[4].in_channel = layer_pars[3].filter_channel;
	layer_pars[4].filter_size = 5;
	layer_pars[4].filter_channel = 64; 
	layer_pars[4].stride = 1;
	layer_pars[4].pad = 2;
	layer_pars[4].padded_in_size = layer_pars[4].in_size + 2 * layer_pars[4].pad;
	layer_pars[4].out_size = (layer_pars[4].padded_in_size - layer_pars[4].filter_size) / layer_pars[4].stride + 1;
	layer_pars[4].minibatch_size = layer_pars[0].minibatch_size;
	layer_pars[4].lr_down_scale = 0.95;

	layer_pars[5].in_size = layer_pars[4].out_size; 
	layer_pars[5].in_channel = layer_pars[4].filter_channel;
	layer_pars[5].filter_channel = layer_pars[4].filter_channel;
	layer_pars[5].pool_size = 3;
	layer_pars[5].stride = 2;
	layer_pars[5].out_size = ceil(((layer_pars[4].out_size - layer_pars[5].pool_size) * 1.0f) \
					 / layer_pars[5].stride) + 1;
	layer_pars[5].minibatch_size = layer_pars[0].minibatch_size;

//	layer_pars[6].w_lr = 1;
//	layer_pars[6].b_lr = 2;
	layer_pars[6].momentum = 0.9;
	layer_pars[6].num_in = layer_pars[5].out_size * layer_pars[5].out_size * layer_pars[5].filter_channel;
	layer_pars[6].num_out = 64;
	layer_pars[6].minibatch_size = layer_pars[0].minibatch_size;
//	layer_pars[6].lr_down_scale = 0.95;

//	layer_pars[7].w_lr = 1;
//	layer_pars[7].b_lr = 2;
	layer_pars[7].momentum = 0.9;
	layer_pars[7].num_in = layer_pars[6].num_out;
	layer_pars[7].num_out = 10;
	layer_pars[7].minibatch_size = layer_pars[0].minibatch_size;
//	layer_pars[7].lr_down_scale = 0.95;



	if(rank == 0){ 
		managerNode(layer_pars);
	}   
	else{
		workerNode(layer_pars);
	} 	

	delete[] layer_pars;
	MPI_Finalize();
	return 0;
}



















