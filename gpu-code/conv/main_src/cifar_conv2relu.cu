/*
 * filename:main.cu
 */

#include <iostream>
#include <fstream>
#include <time.h>
#include <sstream>
#include <cmath>
#include <omp.h>
#include <unistd.h>
#include "mpi.h"
#include "inner_product_layer.hpp"
#include "logistic.hpp"
#include "load_layer.hpp"
#include "param.h"
#include "matrix.hpp"
#include "sigmoid_layer.hpp"
#include "relu_layer.hpp"
#include "convnet.hpp"
#include "pooling_layer.hpp"
#include "matrix.hpp"

using namespace std;

#define THREAD_END 100000
enum swapInfo{SWAP_CNN1_W_PUSH, SWAP_CNN1_BIAS_PUSH, \
	SWAP_CNN2_W_PUSH, SWAP_CNN2_BIAS_PUSH,	\
		SWAP_INNER1_W_PUSH, SWAP_INNER1_BIAS_PUSH, \
		SWAP_SOFTMAX_W_PUSH, SWAP_SOFTMAX_BIAS_PUSH, \
		SWAP_CNN1_W_FETCH, SWAP_CNN1_BIAS_FETCH, \
		SWAP_CNN2_W_FETCH, SWAP_CNN2_BIAS_FETCH, \
		SWAP_INNER1_W_FETCH, SWAP_INNER1_BIAS_FETCH, \
		SWAP_SOFTMAX_W_FETCH, SWAP_SOFTMAX_BIAS_FETCH};

int num_process;
int rank;

int num_train = 50000;
int num_valid = 10000;
int num_minibatch;
int num_validbatch;
int num_train_per_process;
int num_valid_per_process;
int num_epoch = 500;


void managerNode(ConvParam* conv1_cp, ConvParam* conv2_cp, \
		InnerParam* inner1_ip, InnerParam* inner2_ip){


	int cnn1_in_len = conv1_cp->getInSize() * conv1_cp->getInSize() * conv1_cp->getInChannel();
	int cnn1_w_len = conv1_cp->getOutChannel() * conv1_cp->getFilterSize() \
					 * conv1_cp->getFilterSize() * conv1_cp->getInChannel();
	int cnn1_b_len = conv1_cp->getOutChannel();

	int cnn2_w_len = conv2_cp->getOutChannel() * conv2_cp->getFilterSize() \
					 * conv2_cp->getFilterSize() * conv2_cp->getInChannel();
	int cnn2_b_len = conv2_cp->getOutChannel();

	int inner1_w_len = inner1_ip->getNumIn() * inner1_ip->getNumOut();
	int inner1_b_len = inner1_ip->getNumOut();

	int softmax_w_len = inner2_ip->getNumIn() * inner2_ip->getNumOut();
	int softmax_b_len = inner2_ip->getNumOut();

	int train_data_len_part = num_train * cnn1_in_len / (num_process - 1);
	int train_label_len_part = num_train / (num_process - 1);
	int valid_data_len_part = num_valid * cnn1_in_len / (num_process - 1);
	int valid_label_len_part = num_valid / (num_process - 1);

	cout << "done8\n";
	Matrix<float>* train_data = new Matrix<float>(num_train, cnn1_in_len);
	Matrix<float>* valid_data = new Matrix<float>(num_valid, cnn1_in_len);
	Matrix<float>* train_label = new Matrix<float>(num_train, 1);
	Matrix<float>* valid_label = new Matrix<float>(num_valid, 1);

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

	train_data->copyFromHost(cifar10_info->train_pixel, num_train * cnn1_in_len);
	train_label->copyFromHost(cifar10_info->train_label, num_train);
	valid_data->copyFromHost(cifar10_info->test_pixel, num_valid * cnn1_in_len);
	valid_label->copyFromHost(cifar10_info->test_label, num_valid);


	cout << "done6\n";
	Matrix<float>* cnn1_w = new Matrix<float>(conv1_cp->getFilterSize() * \
			conv1_cp->getFilterSize() * conv1_cp->getInChannel(), \
			conv1_cp->getOutChannel());
	Matrix<float>* cnn1_bias = new Matrix<float>(1, conv1_cp->getOutChannel());

	Matrix<float>* cnn2_w = new Matrix<float>(conv2_cp->getFilterSize() * \
			conv2_cp->getFilterSize() * conv2_cp->getInChannel(), \
			conv2_cp->getOutChannel());
	Matrix<float>* cnn2_bias = new Matrix<float>(1, conv2_cp->getOutChannel());

	Matrix<float>* inner1_w = new Matrix<float>(inner1_w_len / inner1_ip->getNumOut(), inner1_ip->getNumOut());
	Matrix<float>* inner1_bias = new Matrix<float>(1, inner1_ip->getNumOut());


	Matrix<float>* softmax_w = new Matrix<float>(softmax_w_len / inner2_ip->getNumOut(), inner2_ip->getNumOut());
	Matrix<float>* softmax_bias = new Matrix<float>(1, inner2_ip->getNumOut());

	cout << "done5\n";
	gaussRand(cnn1_w, 0.001);
	//	initW(cnn1_w);
	gaussRand(cnn2_w, 0.01);
	//	initW(cnn2_w);
	cudaMemset(cnn1_bias->getDevData(), 0, sizeof(float) * cnn1_b_len);
	cudaMemset(cnn2_bias->getDevData(), 0, sizeof(float) * cnn2_b_len);

	gaussRand(inner1_w, 0.01);
	//	initW(inner1_w);
	cudaMemset(inner1_bias->getDevData(), 0, sizeof(float) * inner1_b_len);
	gaussRand(softmax_w, 0.1);
	//initW(softmax_w);
	cudaMemset(softmax_bias->getDevData(), 0, sizeof(float) * softmax_b_len);

	/*
	   readPars(cnn1_w, "cnn1_w_t1.bin");
	   readPars(cnn1_bias, "cnn1_bias_t1.bin");
	   readPars(cnn2_w, "cnn2_w_t1.bin");
	   readPars(cnn2_bias, "cnn2_bias_t1.bin");
	   readPars(inner1_w, "inner1_w_t1.bin");
	   readPars(inner1_bias, "inner1_bias_t1.bin");
	   readPars(softmax_w, "softmax_w_t1.bin");
	   readPars(softmax_bias, "softmax_bias_t1.bin");
	 */

	MPI_Bcast(cnn1_w->getDevData(), cnn1_w_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn1_bias->getDevData(), cnn1_b_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn2_w->getDevData(), cnn2_w_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn2_bias->getDevData(), cnn2_b_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
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
	const int trans_ops = 16;
	const int num_pars_type = 8;
	float* my_pars[num_pars_type] = {cnn1_w->getDevData(), cnn1_bias->getDevData(), \
		cnn2_w->getDevData(), cnn2_bias->getDevData(), \
			inner1_w->getDevData(), inner1_bias->getDevData(), \
			softmax_w->getDevData(), softmax_bias->getDevData()};
	int pars_len[num_pars_type] = {cnn1_w_len, cnn1_b_len, cnn2_w_len, cnn2_b_len, \
		inner1_w_len, inner1_b_len, \
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
	delete inner1_w;
	delete inner1_bias;
	delete softmax_w;
	delete softmax_bias;
}


void workerNode(ConvParam* conv1_cp, FullConnectParam* relu1_fcp, \
		PoolParam* pool1_pp, ConvParam* conv2_cp, FullConnectParam* relu2_fcp, \
		PoolParam* pool2_pp, InnerParam* inner1_ip, FullConnectParam* sigmoid3_fcp, \
		InnerParam* inner2_ip, FullConnectParam* softmax_fcp){
	if(rank == 1){
		cout << "\n===========overall==============" \
			<< "\ntrain: " << num_train \
			<< "\nvalid: " << num_valid \
			<< "\nbatchSize: " << conv1_cp->getMinibatchSize() \
			<< "\nn_fetch: " << conv1_cp->getNFetch() \
			<< "\nn_push: " << conv1_cp->getNPush();

		cout << "\n===========cnn1==============" \
			<< "\nin_size: " << conv1_cp->getInSize() \
			<< "\nin_channel: " << conv1_cp->getInChannel() \
			<< "\nfilter_size: " << conv1_cp->getFilterSize() \
			<< "\nfilter_channel: " << conv1_cp->getOutChannel() \
			<< "\npad: " << conv1_cp->getPad() \
			<< "\nstride: " << conv1_cp->getStride() \
			<< "\nw_lr: " << conv1_cp->getWLR() \
			<< "\nb_lr: " << conv1_cp->getBiasLR() \
			<< "\nmomentum: " << conv1_cp->getMomentum()\
			<< "\nweight_decay: " << conv1_cp->getWeightDecay();

		cout << "\n===========pool1==============" \
			<< "\nin_size: " << pool1_pp->getInSize() \
			<< "\nin_channel: " << pool1_pp->getInChannel() \
			<< "\nfilter_size: " << pool1_pp->getFilterSize() \
			<< "\nfilter_channel: " << pool1_pp->getOutChannel() \
			<< "\npad: " << pool1_pp->getPad() \
			<< "\nstride: " << pool1_pp->getStride();

		cout << "\n===========cnn2==============" \
			<< "\nin_size: " << conv2_cp->getInSize() \
			<< "\nin_channel: " << conv2_cp->getInChannel() \
			<< "\nfilter_size: " << conv2_cp->getFilterSize() \
			<< "\nfilter_channel: " << conv2_cp->getOutChannel() \
			<< "\nstride: " << conv2_cp->getStride() \
			<< "\npad: " << conv2_cp->getPad() \
			<< "\nw_lr: " << conv2_cp->getWLR() \
			<< "\nb_lr: " << conv2_cp->getBiasLR() \
			<< "\nmomentum: " << conv2_cp->getMomentum()\
			<< "\nweight_decay: " << conv2_cp->getWeightDecay();

		cout << "\n===========pool2==============" \
			<< "\nin_size: " << pool2_pp->getInSize() \
			<< "\nin_channel: " << pool2_pp->getInChannel() \
			<< "\nfilter_size: " << pool2_pp->getFilterSize() \
			<< "\nfilter_channel: " << pool2_pp->getOutChannel() \
			<< "\npad: " << pool2_pp->getPad() \
			<< "\nstride: " << pool2_pp->getStride();

		cout << "\n===========inner_product1==============" \
			<< "\nnum_in: " << inner1_ip->getNumIn() \
			<< "\nnum_out: " << inner1_ip->getNumOut() \
			<< "\nw_lr: " << inner1_ip->getWLR() \
			<< "\nb_lr: " << inner1_ip->getBiasLR() \
			<< "\nmomentum: " << inner1_ip->getMomentum() \
			<< "\nweight_decay: " << inner1_ip->getWeightDecay();

		cout << "\n===========softmax==============" \
			<< "\nnum_in: " << inner2_ip->getNumIn() \
			<< "\nnum_out: " << inner2_ip->getNumOut() \
			<< "\nw_lr: " << inner2_ip->getWLR() \
			<< "\nb_lr: " << inner2_ip->getBiasLR() \
			<< "\nmomentum: " << inner2_ip->getMomentum() \
			<< "\nweight_decay: " << inner2_ip->getWeightDecay();
	}

	num_train_per_process = num_train / (num_process - 1);
	num_valid_per_process = num_valid / (num_process - 1);

	int cnn1_in_len = conv1_cp->getInSize() * conv1_cp->getInSize() * conv1_cp->getInChannel();


	int cnn1_w_len = conv1_cp->getOutChannel() * conv1_cp->getFilterSize() \
					 * conv1_cp->getFilterSize() * conv1_cp->getInChannel();
	int cnn1_b_len = conv1_cp->getOutChannel();

	int cnn2_w_len = conv2_cp->getOutChannel() * conv2_cp->getFilterSize() \
					 * conv2_cp->getFilterSize() * conv2_cp->getInChannel();
	int cnn2_b_len = conv2_cp->getOutChannel();

	int inner1_w_len = inner1_ip->getNumIn() * inner1_ip->getNumOut();
	int inner1_b_len = inner1_ip->getNumOut();

	int softmax_w_len = inner2_ip->getNumIn() * inner2_ip->getNumOut();
	int softmax_b_len = inner2_ip->getNumOut();

	int mini_data_len = conv1_cp->getMinibatchSize() * cnn1_in_len;
	int mini_label_len = conv1_cp->getMinibatchSize();

	int train_data_len_part = num_train_per_process * cnn1_in_len;
	int train_label_len_part = num_train_per_process;
	int valid_data_len_part = num_valid_per_process * cnn1_in_len;
	int valid_label_len_part = num_valid_per_process;

	cout << "done4\n";
	ConvNet<float> cnn1(conv1_cp);
	cnn1.initCuda();

	ReluLayer<float> relu1(relu1_fcp);
	relu1.initCuda();

	PoolingLayer<float> pool1(pool1_pp);
	pool1.initCuda();

	ConvNet<float> cnn2(conv2_cp);
	cnn2.initCuda();

	ReluLayer<float> relu2(relu2_fcp);
	relu2.initCuda();

	PoolingLayer<float> pool2(pool2_pp);
	pool2.initCuda();

	InnerProductLayer<float> inner1(inner1_ip);
	inner1.initCuda();

	SigmoidLayer<float> sigmoid3(sigmoid3_fcp);
	sigmoid3.initCuda();

	InnerProductLayer<float> inner2(inner2_ip);
	inner2.initCuda();

	Logistic<float> softmax(softmax_fcp);
	softmax.initCuda();

	Matrix<float>* cnn1_w = cnn1.getW();
	Matrix<float>* cnn1_bias = cnn1.getBias();
	Matrix<float>* cnn2_w = cnn2.getW();
	Matrix<float>* cnn2_bias = cnn2.getBias();
	Matrix<float>* inner1_w = inner1.getW();
	Matrix<float>* inner1_bias = inner1.getBias();
	Matrix<float>* softmax_w = inner2.getW();
	Matrix<float>* softmax_bias = inner2.getBias();

	MPI_Bcast(cnn1_w->getDevData(), cnn1_w_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn1_bias->getDevData(), cnn1_b_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn2_w->getDevData(), cnn2_w_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(cnn2_bias->getDevData(), cnn2_b_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(inner1_w->getDevData(), inner1_w_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(inner1_bias->getDevData(), inner1_b_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(softmax_w->getDevData(), softmax_w_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(softmax_bias->getDevData(), softmax_b_len, MPI_FLOAT, 0, MPI_COMM_WORLD);

	Matrix<float>* train_data = new Matrix<float>(num_train, cnn1_in_len);
	Matrix<float>* train_label = new Matrix<float>(num_train, 1);
	Matrix<float>* valid_data = new Matrix<float>(num_valid, cnn1_in_len);
	Matrix<float>* valid_label = new Matrix<float>(num_valid, 1);

	Matrix<float>* mini_data = new Matrix<float>(conv1_cp->getMinibatchSize(), cnn1_in_len);
	Matrix<float>* mini_label = new Matrix<float>(conv1_cp->getMinibatchSize(), 1);

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

	Matrix<float>* cnn1_y = cnn1.getY();
	Matrix<float>* cnn1_dE_dy = cnn1.getDEDY();
	Matrix<float>* relu1_y = relu1.getY();
	Matrix<float>* relu1_dE_dy = relu1.getDEDY();
	Matrix<float>* pool1_y = pool1.getY();
	Matrix<float>* pool1_dE_dy = pool1.getDEDY();
	Matrix<float>* cnn2_y = cnn2.getY();
	Matrix<float>* cnn2_dE_dy = cnn2.getDEDY();
	Matrix<float>* relu2_y = relu2.getY();
	Matrix<float>* relu2_dE_dy = relu2.getDEDY();
	Matrix<float>* pool2_y = pool2.getY();
	Matrix<float>* pool2_dE_dy = pool2.getDEDY();
	Matrix<float>* inner1_y = inner1.getY();
	Matrix<float>* inner1_dE_dy = inner1.getDEDY();
	Matrix<float>* sigmoid3_y = sigmoid3.getY();
	Matrix<float>* sigmoid3_dE_dy = sigmoid3.getDEDY();
	Matrix<float>* inner2_y = inner2.getY();
	Matrix<float>* inner2_dE_dy = inner2.getDEDY();

	/*
	   cnn1_y->showValue("cnn1_y");
	   cnn2_y->showValue("cnn2_y");
	   relu1_y->showValue("relu1_y");
	   relu2_y->showValue("relu2_y");
	   sigmoid3_y->showValue("sigmoid3_y");
	   inner1_y->showValue("inner1_y");
	   inner2_y->showValue("inner2_y");
	 */

	const int num_pars_type = 8;
	float* my_pars[num_pars_type] = {cnn1_w->getDevData(), cnn1_bias->getDevData(), \
		cnn2_w->getDevData(), cnn2_bias->getDevData(), \
			inner1_w->getDevData(), inner1_bias->getDevData(), \
			softmax_w->getDevData(), softmax_bias->getDevData()};
	int pars_len[num_pars_type] = {cnn1_w_len, cnn1_b_len, cnn2_w_len, cnn2_b_len, \
		inner1_w_len, inner1_b_len, \
			softmax_w_len, softmax_b_len};

	clock_t t;
	t = clock();
	clock_t t1;
	t1 = clock();

	int errorValid;
	float loglihoodValid;
	int totalValid;

	for(int epoch_idx = 0; epoch_idx < num_epoch; epoch_idx++){
		int error = 0;

		for(int batch_idx = 0; batch_idx < num_minibatch; batch_idx++){

			mini_data->changePtrFromStart(train_data->getDevData(), \
					mini_data_len * batch_idx);
			mini_label->changePtrFromStart(train_label->getDevData(), \
					mini_label_len * batch_idx);

			cnn1.computeOutputs(mini_data);
			relu1.computeOutputs(cnn1_y);
			pool1.computeOutputs(relu1_y);

			cnn2.computeOutputs(pool1_y);
			relu2.computeOutputs(cnn2_y);

			pool2.computeOutputs(relu2_y);

			inner1.computeOutputs(pool2_y);
			sigmoid3.computeOutputs(inner1_y);
			inner2.computeOutputs(sigmoid3_y);
			softmax.computeOutputs(inner2_y);

			softmax.computeError(mini_label, error);

			softmax.computeDerivsOfInput(inner2_dE_dy, mini_label);
			inner2.computeDerivsOfPars(sigmoid3_y);
			inner2.computeDerivsOfInput(sigmoid3_dE_dy);

			sigmoid3.computeDerivsOfInput(inner1_dE_dy);
			inner1.computeDerivsOfPars(pool2_y);
			inner1.computeDerivsOfInput(pool2_dE_dy);

			pool2.computeDerivsOfInput(relu2_dE_dy);
			relu2.computeDerivsOfInput(cnn2_dE_dy);
			cnn2.computeDerivsOfPars(cnn2_y);
			cnn2.computeDerivsOfInput(pool1_dE_dy);

			pool1.computeDerivsOfInput(relu1_dE_dy);
			relu1.computeDerivsOfInput(cnn1_dE_dy);
			cnn1.computeDerivsOfPars(mini_data);

			cnn1.updatePars();
			cnn2.updatePars();
			inner1.updatePars();
			inner2.updatePars();

			if((batch_idx + 1) % conv1_cp->getNPush() == 0){
				if(epoch_idx == num_epoch - 1){
					if((batch_idx + conv1_cp->getNPush()) >= num_minibatch \
							|| batch_idx == num_minibatch - 1)
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
			if((batch_idx + 1) % conv1_cp->getNFetch() == 0){
				if(epoch_idx == num_epoch - 1){
					if((batch_idx + conv1_cp->getNFetch()) >= num_minibatch \
							|| batch_idx == num_minibatch - 1)
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

			if(batch_idx == num_minibatch - 1){
				errorValid = 0;
				loglihoodValid = 0.0f;
				for(int validIdx = 0; validIdx < num_validbatch; validIdx++){

					mini_data->changePtrFromStart(valid_data->getDevData(), \
							mini_data_len * validIdx);
					mini_label->changePtrFromStart(valid_label->getDevData(), \
							mini_label_len * validIdx);
					cnn1.computeOutputs(mini_data);
					relu1.computeOutputs(cnn1_y);
					pool1.computeOutputs(relu1_y);

					cnn2.computeOutputs(pool1_y);
					relu2.computeOutputs(cnn2_y);
					pool2.computeOutputs(relu2_y);

					inner1.computeOutputs(pool2_y);
					sigmoid3.computeOutputs(inner1_y);
					inner2.computeOutputs(sigmoid3_y);
					softmax.computeOutputs(inner2_y);

					loglihoodValid += softmax.computeError(mini_label, errorValid);

				}
				totalValid = errorValid;
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
					cout << "epoch_idx: " << epoch_idx << ", accuracy: " \
						<<  1 - (float)totalValid/ num_valid_per_process \
						<< ",likelihood: "<< loglihoodValid<< endl;
			}


		}
		if(rank == 1)
			cout << "train: epoch_idx: " << epoch_idx << ", error: " \
				<<  1 - (float)error/num_train_per_process  << endl;
		if(rank == 1){
			t1 = clock() - t1;
			cout << " " << ((float)t1/CLOCKS_PER_SEC) << " seconds.\n";
			t1 = clock();
		}
		/*
		   if(rank == 1 && epoch_idx % 4 == 0)

		   savePars(cnn1_w, "./pars/cnn1_w_t1.bin");
		   savePars(cnn1_bias, "./pars/cnn1_bias_t1.bin");
		   savePars(cnn2_w, "./pars/cnn2_w_t1.bin");
		   savePars(cnn2_bias, "./pars/cnn2_bias_t1.bin");
		   savePars(inner1_w, "./pars/inner1_w_t1.bin");
		   savePars(inner1_bias, "./pars/inner1_bias_t1.bin");
		   savePars(softmax_w, "./pars/softmax_w_t1.bin");
		   savePars(softmax_bias, "./pars/softmax_bias_t1.bin");
		   }*/

/*
		if((epoch_idx + 1) % 5 == 0){
			conv1_cp->lrMultiScale(0.9);
			conv2_cp->lrMultiScale(0.9);
			inner1_ip->lrMultiScale(0.9);
			inner2_ip->lrMultiScale(0.9);

		}*/

	}
	if(rank == 1){
		t = clock() - t;
		cout << " " << ((float)t/CLOCKS_PER_SEC) / num_epoch << " seconds.\n";
		t = clock();
	}
	savePars(cnn1_w, "./pars/mnist/cnn1_w_t3.bin");
	savePars(cnn1_y, "./pars/mnist/cnn1_y_t3.bin");
	savePars(cnn2_y, "./pars/mnist/cnn2_y_t3.bin");	


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

	int minibatch_size = 100;
	int conv1_in_size = 32;
	int conv1_in_channel = 3;
	int conv1_pad = 2;
	int conv1_stride = 1;
	int conv1_filter_size = 5;
	int conv1_out_channel = 16;
	float conv1_w_lr = 0.001;
	float conv1_b_lr = 0.002;
	float conv1_momentum = 0.9;
	float conv1_weight_decay = 0.004;
	int n_push = 49;
	int n_fetch = 50;

	//sigmoid层参数可以先忽略
	int pool1_pad = 0;
	int pool1_stride = 2;
	int pool1_filter_size = 3;
	PoolingType pool1_type = MAX_POOLING;

	int conv2_pad = 2;
	int conv2_stride = 1;
	int conv2_filter_size = 5;
	int conv2_out_channel = 32;
	float conv2_w_lr = 0.001;
	float conv2_b_lr = 0.002;
	float conv2_momentum = 0.9;
	float conv2_weight_decay = 0.004;

	int pool2_pad = 0;
	int pool2_stride = 2;
	int pool2_filter_size = 3;
	PoolingType pool2_type = MAX_POOLING;

	int inner1_num_out = 1000;
	float inner1_w_lr = 0.001;
	float inner1_b_lr = 0.002;
	float inner1_momentum = 0.9;
	float inner1_weight_decay = 0.004;

	int inner2_num_out = 10;
	float inner2_w_lr = 0.001;
	float inner2_b_lr = 0.002;
	float inner2_momentum = 0.9;
	float inner2_weight_decay = 0.004;

	ConvParam* conv1_cp = new ConvParam("conv1_layer", minibatch_size, \
			conv1_w_lr, conv1_b_lr, conv1_momentum, conv1_weight_decay, \
			n_push, n_fetch, conv1_in_size, conv1_pad, conv1_stride, \
			conv1_in_channel, conv1_filter_size, conv1_out_channel);

	FullConnectParam* relu1_fcp = new FullConnectParam("relu1_layer", \
			0, conv1_cp);

	PoolParam* pool1_pp = new PoolParam("pool1_layer", pool1_pad, \
			pool1_stride, pool1_filter_size, 0, conv1_cp, pool1_type);

	ConvParam* conv2_cp = new ConvParam("conv2_layer", conv2_w_lr, \
			conv2_b_lr, conv2_momentum, conv2_weight_decay, n_push, \
			n_fetch, conv2_pad, conv2_stride, conv2_filter_size, \
			conv2_out_channel, pool1_pp);

	FullConnectParam* relu2_fcp = new FullConnectParam("relu2_layer", \
			0, conv2_cp);

	PoolParam* pool2_pp = new PoolParam("pool2_layer", pool2_pad, \
			pool2_stride, pool2_filter_size, 0, conv2_cp, pool2_type);

	InnerParam* inner1_ip = new InnerParam("inner1_layer", inner1_w_lr, \
			inner1_b_lr, inner1_momentum, inner1_weight_decay, n_push, \
			n_fetch, inner1_num_out, pool2_pp);

	FullConnectParam* sigmoid3_y = new FullConnectParam("sigmoid3_layer", \
			0, inner1_ip);

	InnerParam* inner2_ip = new InnerParam("inner2_layer", inner2_w_lr, \
			inner2_b_lr, inner2_momentum, inner2_weight_decay, \
			n_push, n_fetch, inner2_num_out, sigmoid3_y);

	FullConnectParam* softmax_fcp = new FullConnectParam("softmax_layer", \
			0, inner2_ip);


	num_minibatch = num_train / (minibatch_size * (num_process - 1));
	num_validbatch = num_valid / (minibatch_size * (num_process - 1));
	if(rank == 0){ 
		managerNode(conv1_cp, conv2_cp, inner1_ip, inner2_ip);
	}   
	else{
		workerNode(conv1_cp, relu1_fcp, pool1_pp, \
				conv2_cp, relu2_fcp, pool2_pp, \
				inner1_ip, sigmoid3_y, \
				inner2_ip, softmax_fcp);
	} 	

	MPI_Finalize();
	return 0;
}



















