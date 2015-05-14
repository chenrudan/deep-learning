/*
 * filename:main.cu
 */

#include <iostream>
#include <fstream>
#include <time.h>
#include <sstream>
#include <cmath>
#include <omp.h>
#include "mpi.h"
#include "matrix.h"
#include "nvmatrix.cuh"
#include "inner_product_layer.cuh"
#include "utils.cuh"
#include "logistic.cuh"
#include "load_layer.hpp"
#include "layer_kernel.cuh"

using namespace std;

#define THREAD_END 100000
enum swapInfo{SWAP_INNER1_W_PUSH, SWAP_INNER1_BIAS_PUSH, \
	SWAP_SOFTMAX_W_PUSH, SWAP_SOFTMAX_BIAS_PUSH, \
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

	cout << "\n===========inner_product1==============" \
		<< "\nnum_in: " << layer_pars[0].num_in \
		<< "\nnum_out: " << layer_pars[0].num_out \
		<< "\nw_lr: " << layer_pars[0].w_lr \
		<< "\nb_lr: " << layer_pars[0].b_lr \
		<< "\nmomentum: " << layer_pars[0].momentum \
		<< "\nweight_decay: " << layer_pars[0].weight_decay \
		<< "\nlr_scale: " << layer_pars[0].lr_down_scale;

	cout << "\n===========softmax==============" \
		<< "\nnum_in: " << layer_pars[1].num_in \
		<< "\nnum_out: " << layer_pars[1].num_out \
		<< "\nw_lr: " << layer_pars[1].w_lr \
		<< "\nb_lr: " << layer_pars[1].b_lr \
		<< "\nmomentum: " << layer_pars[1].momentum \
		<< "\nweight_decay: " << layer_pars[1].weight_decay \
		<< "\nlr_scale: " << layer_pars[1].lr_down_scale << endl;


	int inner1_in_len = layer_pars[0].in_size * layer_pars[0].in_size * layer_pars[0].in_channel;

	int inner1_w_len = layer_pars[0].num_in * layer_pars[0].num_out;
	int inner1_b_len = layer_pars[0].num_out;

	int softmax_w_len = layer_pars[1].num_in * layer_pars[1].num_out;
	int softmax_b_len = layer_pars[1].num_out;

	int train_data_len_part = layer_pars[0].num_train * inner1_in_len / (num_process - 1);
	int train_label_len_part = layer_pars[0].num_train / (num_process - 1);
	int valid_data_len_part = layer_pars[0].num_valid * inner1_in_len / (num_process - 1);
	int valid_label_len_part = layer_pars[0].num_valid / (num_process - 1);

cout << "done8\n";
	NVMatrix* train_data = new NVMatrix(layer_pars[0].num_train, inner1_in_len);
	NVMatrix* valid_data = new NVMatrix(layer_pars[0].num_valid, inner1_in_len);
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
        ss << 5;
        ss >> s;    
		string filename = "../data/cifar-10-batches-bin/data_batch_"+s+".bin";
        cifar10.loadBinary(filename, cifar10_info->train_pixel_ptr, \
				cifar10_info->train_label_ptr);    
    }   
    cifar10.loadBinary("../data/cifar-10-batches-bin/test_batch.bin", \
            cifar10_info->test_pixel_ptr, cifar10_info->test_label_ptr);

	train_data->copyFromHost(cifar10_info->train_pixel, layer_pars[0].num_train * inner1_in_len);
	train_label->copyFromHost(cifar10_info->train_label, layer_pars[0].num_train);
	valid_data->copyFromHost(cifar10_info->test_pixel, layer_pars[0].num_valid * inner1_in_len);
	valid_label->copyFromHost(cifar10_info->test_label, layer_pars[0].num_valid);


cout << "done6\n";

	NVMatrix* inner1_w = new NVMatrix(inner1_w_len / layer_pars[0].num_out, layer_pars[0].num_out);
	NVMatrix* inner1_bias = new NVMatrix(1, layer_pars[0].num_out);


	NVMatrix* softmax_w = new NVMatrix(softmax_w_len / layer_pars[1].num_out, layer_pars[1].num_out);
	NVMatrix* softmax_bias = new NVMatrix(1, layer_pars[1].num_out);

cout << "done5\n";
//	gaussRand(inner1_w, 0.01);
	initW(inner1_w);
	cudaMemset(inner1_bias->getDevData(), 0, sizeof(float) * inner1_b_len);
//	gaussRand(softmax_w, 0.01);
	initW(softmax_w);
	cudaMemset(softmax_bias->getDevData(), 0, sizeof(float) * softmax_b_len);

	//	readPars(hHidVis, "hHidVis_t1.bin");
	//	readPars(hHidBiases, "hHidBiases_t1.bin");
	//	readPars(hsoftmax_w, "hsoftmax_w_t1.bin");
	//	readPars(hsoftmax_bias, "hsoftmax_bias_t1.bin");

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

	//pro进程，每个进程进行的数据交换次数，0123是push，4567是fetch
	//4个数据地址，8个线程来分别实现两种操作
	const int trans_ops = 8;
	const int num_pars_type = 4;
	float* my_pars[num_pars_type] = {inner1_w->getDevData(), inner1_bias->getDevData(), \
			softmax_w->getDevData(), softmax_bias->getDevData()};
	int pars_len[num_pars_type] = {inner1_w_len, inner1_b_len, \
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

	delete cifar10_info;
	delete train_data;
	delete train_label;
	delete valid_data;
	delete valid_label;
	delete inner1_w;
	delete inner1_bias;
	delete softmax_w;
	delete softmax_bias;
}


void workerNode(pars* layer_pars){
	
	layer_pars[0].num_train /= (num_process - 1);
	int inner1_in_len = layer_pars[0].in_size * layer_pars[0].in_size * layer_pars[0].in_channel;

	int inner1_w_len = layer_pars[0].num_in * layer_pars[0].num_out;
	int inner1_b_len = layer_pars[0].num_out;

	int softmax_w_len = layer_pars[1].num_in * layer_pars[1].num_out;
	int softmax_b_len = layer_pars[1].num_out;

	int mini_data_len = layer_pars->minibatch_size * inner1_in_len;
	int mini_label_len = layer_pars->minibatch_size;

	int train_data_len_part = layer_pars->num_train * inner1_in_len;
	int train_label_len_part = layer_pars->num_train;
	int valid_data_len_part = layer_pars->num_valid * inner1_in_len;
	int valid_label_len_part = layer_pars->num_valid;

cout << "done4\n";

	InnerProductLayer inner1(layer_pars);
	inner1.initCuda();

	Logistic softmax1(layer_pars + 1);
	softmax1.initCuda();


	NVMatrix* inner1_w = inner1.getW();
	NVMatrix* inner1_bias = inner1.getBias();
	NVMatrix* softmax_w = softmax1.getW();
	NVMatrix* softmax_bias = softmax1.getBias();

	MPI_Bcast(inner1_w->getDevData(), inner1_w_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(inner1_bias->getDevData(), inner1_b_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(softmax_w->getDevData(), softmax_w_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(softmax_bias->getDevData(), softmax_b_len, MPI_FLOAT, 0, MPI_COMM_WORLD);

	NVMatrix* train_data = new NVMatrix(layer_pars->num_train, inner1_in_len);
	NVMatrix* train_label = new NVMatrix(layer_pars->num_train, 1);
	NVMatrix* valid_data = new NVMatrix(layer_pars->num_valid, inner1_in_len);
	NVMatrix* valid_label = new NVMatrix(layer_pars->num_valid, 1);

	NVMatrix* mini_data = new NVMatrix(layer_pars->minibatch_size, inner1_in_len);
	NVMatrix* mini_label = new NVMatrix(layer_pars->minibatch_size, 1);

cout << "done1\n";
	MPI_Status status;
	MPI_Recv(train_data->getDevData(), train_data_len_part, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);
	MPI_Recv(train_label->getDevData(), train_label_len_part, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);
	MPI_Recv(valid_data->getDevData(), valid_data_len_part, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);
	MPI_Recv(valid_label->getDevData(), valid_label_len_part, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);

cout << "done2\n";
	int passMsg = 0;

	NVMatrix* inner1_y;
	NVMatrix* inner1_dE_dy;

	const int num_pars_type = 4;
	float* my_pars[num_pars_type] = {inner1_w->getDevData(), inner1_bias->getDevData(), \
			softmax_w->getDevData(), softmax_bias->getDevData()};
	int pars_len[num_pars_type] = {inner1_w_len, inner1_b_len, \
			     softmax_w_len, softmax_b_len};

	clock_t t;
	t = clock();
	clock_t t1;
	t1 = clock();
	for(int epoch_idx = 0; epoch_idx < layer_pars->num_epoch; epoch_idx++){
		int error = 0;
/*
if(epoch_idx > 1){
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

			inner1.computeOutputs(mini_data);
			inner1_y = inner1.getY();
			softmax1.computeOutputs(inner1_y);
			softmax1.computeError(mini_label, error);

			softmax1.computeDerivsOfPars(inner1_y, mini_label);
			inner1_dE_dy = inner1.getDEDY();
			softmax1.computeDerivsOfInput(inner1_dE_dy);
			inner1.computeDerivsOfPars(mini_data);

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
					inner1.computeOutputs(mini_data);
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
		
		if((epoch_idx + 1) % 4){
			inner1.transfarLowerPars();
			softmax1.transfarLowerPars();
		} 


	}
	if(rank == 1){
		t = clock() - t;
		cout << " " << ((float)t/CLOCKS_PER_SEC) / layer_pars->num_epoch << " seconds.\n";
		t = clock();
	}

	delete mini_data;
	delete mini_label;
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

	const int num_layer = 2;

	pars* layer_pars = new pars[num_layer];

	layer_pars[0].num_train = 50000;
	layer_pars[0].num_valid = 10000;
	layer_pars[0].minibatch_size = 100;
	layer_pars[0].num_minibatch = layer_pars[0].num_train / (layer_pars[0].minibatch_size * (num_process - 1));
	layer_pars[0].num_validbatch = layer_pars[0].num_valid / (layer_pars[0].minibatch_size * (num_process - 1));
	layer_pars[0].num_epoch = 500; 
	layer_pars[0].n_push = 49;
	layer_pars[0].n_fetch = 50;
	layer_pars[0].in_size = 32;
	layer_pars[0].in_channel = 3;

	layer_pars[0].w_lr = 0.002;
	layer_pars[0].b_lr = 0.002;
	layer_pars[0].momentum = 0.9;
	layer_pars[0].weight_decay = 0;
	layer_pars[0].num_in = 32*32*3;
	layer_pars[0].num_out = 1000;
	layer_pars[0].lr_down_scale = 0.95;

	layer_pars[1].w_lr = 0.0001;
	layer_pars[1].b_lr = 0.0001;
	layer_pars[1].momentum = 0.9;
	layer_pars[1].weight_decay = 0;
	layer_pars[1].num_in = layer_pars[0].num_out;
	layer_pars[1].num_out = 10;
	layer_pars[1].minibatch_size = layer_pars[0].minibatch_size;
	layer_pars[1].lr_down_scale = 0.95;



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



















