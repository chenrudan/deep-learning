/*
 * filename:main.cu
 */

#include <iostream>
#include <fstream>
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

using namespace std;

#define THREAD_END 100000
enum swapInfo{SWAP_INNER1_W_PUSH, SWAP_INNER1_BIAS_PUSH, \
	SWAP_SOFTMAX_W_PUSH, SWAP_SOFTMAX_BIAS_PUSH, \
	SWAP_INNER1_W_FETCH, SWAP_INNER1_BIAS_FETCH, \
	SWAP_SOFTMAX_W_FETCH, SWAP_SOFTMAX_BIAS_FETCH};

int num_process;
int rank;

int num_train = 50000;
int num_train_per_process = 0;
int num_valid = 10000;
int num_valid_per_process = 0;
int num_minibatch = 0;
int num_validbatch = 0;
int num_epoch = 500;

void managerNode(InnerParam* inner1_fcp, InnerParam* softmax1_fcp){

	cout << "\n===========overall==============" \
		<< "\ntrain: " << num_train \
		<< "\nvalid: " << num_valid \
		<< "\nbatchSize: " << inner1_fcp->getMinibatchSize() \
		<< "\nn_fetch: " << inner1_fcp->getNFetch() \
		<< "\nn_push: " << inner1_fcp->getNPush();

	cout << "\n===========inner_product1==============" \
		<< "\nnum_in: " << inner1_fcp->getNumIn() \
		<< "\nnum_out: " << inner1_fcp->getNumOut() \
		<< "\nw_lr: " << inner1_fcp->getWLR() \
		<< "\nb_lr: " << inner1_fcp->getBiasLR() \
		<< "\nmomentum: " << inner1_fcp->getMomentum();

	cout << "\n===========softmax==============" \
		<< "\nnum_in: " << softmax1_fcp->getNumIn() \
		<< "\nnum_out: " << softmax1_fcp->getNumOut() \
		<< "\nw_lr: " << softmax1_fcp->getWLR() \
		<< "\nb_lr: " << softmax1_fcp->getBiasLR() \
		<< "\nmomentum: " << softmax1_fcp->getMomentum() << endl;


	int inner1_in_len = inner1_fcp->getNumIn();

	int inner1_w_len = inner1_fcp->getNumIn() * inner1_fcp->getNumOut();
	int inner1_b_len = inner1_fcp->getNumOut();

	int softmax_w_len = softmax1_fcp->getNumIn() * softmax1_fcp->getNumOut();
	int softmax_b_len = softmax1_fcp->getNumOut();

	int train_data_len_part = num_train * inner1_in_len / (num_process - 1);
	int train_label_len_part = num_train / (num_process - 1);
	int valid_data_len_part = num_valid * inner1_in_len / (num_process - 1);
	int valid_label_len_part = num_valid / (num_process - 1);

cout << "done8\n";
	Matrix<float>* train_data = new Matrix<float>(num_train, inner1_in_len);
	Matrix<float>* valid_data = new Matrix<float>(num_valid, inner1_in_len);
	Matrix<float>* train_label = new Matrix<float>(num_train, 1);
	Matrix<float>* valid_label = new Matrix<float>(num_valid, 1);


    readData(train_data, "../data/input/mnist_train.bin", true);
    readData(valid_data, "../data/input/mnist_valid.bin", true);
    readData(train_label, "../data/input/mnist_label_train.bin", false);
    readData(valid_label, "../data/input/mnist_label_valid.bin", false);


cout << "done7\n";

	ImgInfo<float> *cifar10_info = new ImgInfo<float>;
/*	LoadCifar10<float> cifar10(cifar10_info);
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

	train_data->copyFromHost(cifar10_info->train_pixel, num_train * inner1_in_len);
	train_label->copyFromHost(cifar10_info->train_label, num_train);
	valid_data->copyFromHost(cifar10_info->test_pixel, num_valid * inner1_in_len);
	valid_label->copyFromHost(cifar10_info->test_label, num_valid);

*/
cout << "done6\n";

	Matrix<float>* inner1_w = new Matrix<float>(inner1_w_len / inner1_fcp->getNumOut(), inner1_fcp->getNumOut());
	Matrix<float>* inner1_bias = new Matrix<float>(1, inner1_fcp->getNumOut());


	Matrix<float>* softmax_w = new Matrix<float>(softmax_w_len / softmax1_fcp->getNumOut(), softmax1_fcp->getNumOut());
	Matrix<float>* softmax_bias = new Matrix<float>(1, softmax1_fcp->getNumOut());

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


void workerNode(InnerParam* inner1_fcp, FullConnectParam* sigmoid1_fcp, InnerParam* softmax1_fcp){

	num_train_per_process = num_train / (num_process - 1);
	num_valid_per_process = num_valid / (num_process - 1);
	int inner1_in_len = inner1_fcp->getNumIn();

	int inner1_w_len = inner1_fcp->getNumIn() * inner1_fcp->getNumOut();
	int inner1_b_len = inner1_fcp->getNumOut();

	int softmax_w_len = softmax1_fcp->getNumIn() * softmax1_fcp->getNumOut();
	int softmax_b_len = softmax1_fcp->getNumOut();

	int mini_data_len = inner1_fcp->getMinibatchSize()  * inner1_in_len;
	int mini_label_len = inner1_fcp->getMinibatchSize() ;

	int train_data_len_part = num_train_per_process * inner1_in_len;
	int train_label_len_part = num_train_per_process;
	int valid_data_len_part = num_valid_per_process * inner1_in_len;
	int valid_label_len_part = num_valid_per_process;

cout << "done4\n";

	InnerProductLayer<float> inner1(inner1_fcp);
	inner1.initCuda();

	SigmoidLayer<float> sigmoid1(sigmoid1_fcp);
	sigmoid1.initCuda();

	Logistic<float> softmax1(softmax1_fcp);
	softmax1.initCuda();

	Matrix<float>* inner1_w = inner1.getW();
	Matrix<float>* inner1_bias = inner1.getBias();
	Matrix<float>* softmax_w = softmax1.getW();
	Matrix<float>* softmax_bias = softmax1.getBias();

	MPI_Bcast(inner1_w->getDevData(), inner1_w_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(inner1_bias->getDevData(), inner1_b_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(softmax_w->getDevData(), softmax_w_len, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(softmax_bias->getDevData(), softmax_b_len, MPI_FLOAT, 0, MPI_COMM_WORLD);

	Matrix<float>* train_data = new Matrix<float>(num_train_per_process, inner1_in_len);
	Matrix<float>* train_label = new Matrix<float>(num_train_per_process, 1);
	Matrix<float>* valid_data = new Matrix<float>(num_valid_per_process, inner1_in_len);
	Matrix<float>* valid_label = new Matrix<float>(num_valid_per_process, 1);

	Matrix<float>* mini_data = new Matrix<float>(inner1_fcp->getMinibatchSize() , inner1_in_len);
	Matrix<float>* mini_label = new Matrix<float>(inner1_fcp->getMinibatchSize() , 1);

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

	Matrix<float>* inner1_y;
	Matrix<float>* inner1_dE_dy;
	Matrix<float>* sigmoid1_y;
	Matrix<float>* sigmoid1_dE_dy;

	const int num_pars_type = 4;
	float* my_pars[num_pars_type] = {inner1_w->getDevData(), inner1_bias->getDevData(), \
			softmax_w->getDevData(), softmax_bias->getDevData()};
	int pars_len[num_pars_type] = {inner1_w_len, inner1_b_len, \
			     softmax_w_len, softmax_b_len};

	clock_t t;
	t = clock();
	clock_t t1;
	t1 = clock();
	for(int epoch_idx = 0; epoch_idx < num_epoch; epoch_idx++){
		int error = 0;
/*
if(epoch_idx > 1){
	inner1_w->showValue("inner1w");
	inner1_bias->showValue("inner1b");
	softmax_w->showValue("softmaxw");
	softmax_bias->showValue("softmaxb");
}
*/
		for(int batch_idx = 0; batch_idx < num_minibatch; batch_idx++){

			mini_data->changePtrFromStart(train_data->getDevData(), \
					mini_data_len * batch_idx);
			mini_label->changePtrFromStart(train_label->getDevData(), \
					mini_label_len * batch_idx);

			inner1.computeOutputs(mini_data);
			inner1_y = inner1.getY();
			sigmoid1.computeOutputs(inner1_y);
			sigmoid1_y = sigmoid1.getY();
			softmax1.computeOutputs(sigmoid1_y);
			softmax1.computeError(mini_label, error);

			softmax1.computeDerivsOfPars(sigmoid1_y, mini_label);
			sigmoid1_dE_dy = sigmoid1.getDEDY();
			softmax1.computeDerivsOfInput(sigmoid1_dE_dy);
			inner1_dE_dy = inner1.getDEDY();
			sigmoid1.computeDerivsOfInput(inner1_dE_dy);
			inner1.computeDerivsOfPars(mini_data);

			inner1.updatePars();
			softmax1.updatePars();

			if((batch_idx + 1) % inner1_fcp->getNPush()  == 0){
				if(epoch_idx == num_epoch - 1){
					if((batch_idx + inner1_fcp->getNPush() ) >= num_minibatch \
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
			if((batch_idx + 1) % inner1_fcp->getNFetch()  == 0){
				if(epoch_idx == num_epoch - 1){
					if((batch_idx + inner1_fcp->getNFetch() ) >= num_minibatch \
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
				int errorValid = 0;
				float loglihoodValid = 0;
				for(int validIdx = 0; validIdx < num_validbatch; validIdx++){

					mini_data->changePtrFromStart(valid_data->getDevData(), \
							mini_data_len * validIdx);
					mini_label->changePtrFromStart(valid_label->getDevData(), \
							mini_label_len * validIdx);
					inner1.computeOutputs(mini_data);
					inner1_y = inner1.getY();
					sigmoid1.computeOutputs(inner1_y);
					sigmoid1_y = sigmoid1.getY();
					softmax1.computeOutputs(sigmoid1_y);
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
						<<  (float)totalValid/num_valid \
						<< ",likelihood: "<< loglihoodValid<< endl;
			}
		}
		
		if(rank == 1){
			t1 = clock() - t1;
			cout << " " << ((float)t1/CLOCKS_PER_SEC) << " seconds.\n";
			t1 = clock();
		}
		/*
		if((epoch_idx + 1) % 4 == 0 ){
			inner1_fcp->lrMultiScale(0.95);
			softmax1_fcp->lrMultiScale(0.95);
		}*/ 


	}
	if(rank == 1){
		t = clock() - t;
		cout << " " << ((float)t/CLOCKS_PER_SEC) / num_epoch << " seconds.\n";
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

	int minibatch_size = 100;
	int inner1_num_in = 28*28;
	int inner1_num_out = 500;
	int softmax_num_out = 10;
	float inner1_w_lr = 0.1;
	float inner1_b_lr = 0.1;
	float inner1_momentum = 0.9;
	float inner1_weight_decay = 0;
	int n_push = 50;
	int n_fetch = 49;
	float softmax_w_lr = 0.1;
	float softmax_b_lr = 0.1;
	float softmax_momentum = 0.9;
	float softmax_weight_decay = 0;

	InnerParam* inner1_fcp = new InnerParam("inner_product_layer1", \
			minibatch_size, inner1_w_lr, inner1_b_lr, \
			inner1_momentum, inner1_weight_decay, n_push, n_fetch, \
			inner1_num_in, inner1_num_out);

	FullConnectParam* sigmoid1_fcp = new FullConnectParam("sigmoid_layer", \
			minibatch_size, inner1_num_out, inner1_num_out);

	InnerParam* softmax_fcp = new InnerParam("softmax_layer1", \
			softmax_w_lr, softmax_b_lr, softmax_momentum, \
			softmax_weight_decay, n_push, n_fetch, softmax_num_out, inner1_fcp);
/*
cout << inner1_fcp->getWLR() << ":" \
	<< inner1_fcp->getNPush() << ":" \
	<< inner1_fcp->getNFetch() << ":" \
	<< inner1_fcp->getNumIn() << ":" \
	<< inner1_fcp->getNumOut() << ":" \
	<< inner1_fcp->getMinibatchSize() << "\n"; 

cout << softmax_fcp->getWLR() << ":" \
	<< softmax_fcp->getNPush() << ":" \
	<< softmax_fcp->getNFetch() << ":" \
	<< softmax_fcp->getNumIn() << ":" \
	<< softmax_fcp->getNumOut() << ":" \
	<< softmax_fcp->getMinibatchSize() << "\n"; 
*/
	num_minibatch = num_train / (minibatch_size * (num_process - 1));
	num_validbatch = num_valid / (minibatch_size * (num_process - 1));

	if(rank == 0){ 
		managerNode(inner1_fcp, softmax_fcp);
	}   
	else{
		workerNode(inner1_fcp, sigmoid1_fcp, softmax_fcp);
	} 	

	MPI_Finalize();
	return 0;
}



















