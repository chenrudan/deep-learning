/*
 * filename:conv3.cu
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <omp.h>
#include <vector>
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
#include "dropout_layer.hpp"
#include "model_component.hpp"

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

int num_train_per_process;
int num_valid_per_process;

void managerNode(ModelComponent* model_component){
	//首先处理数据
	cout << "Loading data...\n";
	LoadCifar10<float>* cifar10 = new LoadCifar10<float>(50000, 10000, 0, 32, 3);
	int num_train = 50000;
	int num_valid = 10000;
	model_component->setNumTrain(num_train);
	model_component->setNumValid(num_valid);

	int train_data_len_part = num_train*model_component->getInLen()/(num_process-1);
	int train_label_len_part = num_train / (num_process - 1);
	int valid_data_len_part = num_valid*model_component->getInLen()/(num_process-1);
	int valid_label_len_part = num_valid / (num_process - 1);

	Matrix<float>* train_data = new Matrix<float>(num_train, model_component->getInLen());
	Matrix<float>* valid_data = new Matrix<float>(num_valid, model_component->getInLen());
	Matrix<float>* train_label = new Matrix<float>(num_train, 1);
	Matrix<float>* valid_label = new Matrix<float>(num_valid, 1);

/*
    readData(train_data, "../data/input/mnist_train.bin", true);
    readData(valid_data, "../data/input/mnist_valid.bin", true);
    readData(train_label, "../data/input/mnist_label_train.bin", false);
    readData(valid_label, "../data/input/mnist_label_valid.bin", false);

*/
	train_data->copyFromHost(cifar10->getTrainPixel(), \
	                   num_train*model_component->getInLen());
	train_label->copyFromHost(cifar10->getTrainLabel(), num_train);
	valid_data->copyFromHost(cifar10->getValidPixel(), \
	                   num_valid*model_component->getInLen());
	valid_label->copyFromHost(cifar10->getValidLabel(), num_valid);

	delete cifar10;

//	savePars(valid_data, "./snapshot/input_snap/valid_data.bin");
//	savePars(valid_label, "./snapshot/input_snap/valid_label.bin");

	cout << "Loading data is done.\n";

	Matrix<float> *w[model_component->getNumNeedTrainLayers()];
	Matrix<float> *bias[model_component->getNumNeedTrainLayers()];

	vector<Param*> train_param = model_component->getNeedTrainLayersParam();

	for (int j = 0; j < model_component->getNumNeedTrainLocalLayers(); ++j) {
		w[j] = new Matrix<float>(train_param[j]->getFilterSize() * \
			train_param[j]->getFilterSize() * train_param[j]->getInChannel(), \
			train_param[j]->getOutChannel());
		bias[j] = new Matrix<float>(1, train_param[j]->getOutChannel());
	}
	for (int j = model_component->getNumNeedTrainLocalLayers(); \
			j < model_component->getNumNeedTrainLayers(); ++j) {
		w[j] = new Matrix<float>(train_param[j]->getNumIn(), \
					train_param[j]->getNumOut());
		bias[j] = new Matrix<float>(1, train_param[j]->getNumOut());
	}

	cout << "Initialize weight and bias...\n";
	gaussRand(w[0], 0.001);
	gaussRand(w[1], 0.01);
	gaussRand(w[2], 0.01);
	cudaMemset(bias[0]->getDevData(), 0, sizeof(float)*model_component->getBiasLen()[0]);
	cudaMemset(bias[1]->getDevData(), 0, sizeof(float)*model_component->getBiasLen()[1]);
	cudaMemset(bias[2]->getDevData(), 0, sizeof(float)*model_component->getBiasLen()[2]);
	gaussRand(w[3], 0.1);
	cudaMemset(bias[3]->getDevData(), 0, sizeof(float)*model_component->getBiasLen()[3]);
	gaussRand(w[4], 0.1);
	cudaMemset(bias[4]->getDevData(), 0, sizeof(float)*model_component->getBiasLen()[4]);

	for (int k = 0; k < model_component->getNumNeedTrainLayers(); ++k) {
		MPI_Bcast(w[k]->getDevData(), model_component->getWLen()[k], \
		           MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(bias[k]->getDevData(), model_component->getBiasLen()[k], \
		           MPI_FLOAT, 0, MPI_COMM_WORLD);
	}

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
	delete train_data;
	delete train_label;
	delete valid_data;
	delete valid_label;

	//pro进程，每个进程进行的数据交换次数
	const int trans_ops = 2*2*model_component->getNumNeedTrainLayers();

	const int num_pars_type = 2*model_component->getNumNeedTrainLayers();

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
				#pragma omp parallel num_threads(2)
				{
					int t_tid = omp_get_thread_num();
					if (t_tid == 0) {
						MPI_Recv(w[pars_addr], model_component->getWLen()[pars_addr], \
				               MPI_FLOAT, pid, swap_id + par_state, \
						       MPI_COMM_WORLD, &status);
					}else {
						MPI_Recv(bias[pars_addr], model_component->getBiasLen()[pars_addr], \
				               MPI_FLOAT, pid, swap_id + par_state, \
						       MPI_COMM_WORLD, &status);
					}
				}
			}else{
				#pragma omp parallel num_threads(2)
				{
					int t_tid = omp_get_thread_num();
					if (t_tid == 0) {
						MPI_Send(w[pars_addr], model_component->getWLen()[pars_addr], \
				               MPI_FLOAT, pid, swap_id + par_state, \
						       MPI_COMM_WORLD);
					}else {
						MPI_Send(bias[pars_addr], model_component->getBiasLen()[pars_addr], \
				               MPI_FLOAT, pid, swap_id + par_state, \
						       MPI_COMM_WORLD);
					}
				}
			}   
		}
	}
}


void workerNode(ModelComponent* model_component){

	if(rank == 1){
		vector<Param*> layers_param = model_component->getLayersParamPtr();

		cout << "\n===========overall==============" \
			<< "\ntrain: " << model_component->getNumTrain() \
			<< "\nvalid: " << model_component->getNumvalid() \
			<< "\nbatchSize: " << layers_param->getMinibatchSize() \
			<< "\nn_fetch: " << layers_param->getNFetch() \
			<< "\nn_push: " << layers_param->getNPush();

		for (int i = 0; i < model_component->getNumLocalLayers(); ++i) {
			cout << "\n============"<< layers_param[i]->getName() << "============" \
				<< "\nin_size: " << layers_param[i]->getInSize() \
				<< "\nin_channel: " << layers_param[i]->getInChannel() \
				<< "\nfilter_size: " << layers_param[i]->getFilterSize() \
				<< "\nfilter_channel: " << layers_param[i]->getOutChannel() \
				<< "\npad: " << layers_param[i]->getPad() \
				<< "\nstride: " << layers_param[i]->getStride();
		}
		for (int i = model_component->getNumLocalLayers(); \
					i < model_component->getNumLayers(); ++i) {
			cout << "\n============"<< layers_param[i]->getName() << "============" \
				<< "\nnum_in: " << layers_param[i]->getNumIn() \
				<< "\nnum_out: " << layers_param[i]->getNumOut();
		}
		for (int i = 0; i < model_component->getNumNeedTrainLayers(); ++i) {
			cout << "\n============"<< layers_param[i]->getName() << "============" \
 				<< "\nw_lr: " << layers_param[i]->getWLR() \
				<< "\nb_lr: " << layers_param[i]->getBiasLR() \
				<< "\nmomentum: " << layers_param[i]->getMomentum()\
				<< "\nweight_decay: " << layers_param[i]->getWeightDecay();
		}
	}

	cout << "Initialize layers...\n";

	Layer<float> *layers[model_component->getNumLayers()];
	
	layers[0] = new ConvNet<float>(layers_param[0]);
	layers[1] = new ReluLayer<float>(layers_param[1]);
	layers[2] = new PoolingLayer<float>(layers_param[2]);
	layers[3] = new ConvNet<float>(layers_param[3]);
	layers[4] = new ReluLayer<float>(layers_param[4]);
	layers[5] = new PoolingLayer<float>(layers_param[5]);
	layers[6] = new ConvNet<float>(layers_param[6]);
	layers[7] = new ReluLayer<float>(layers_param[7]);
	layers[8] = new PoolingLayer<float>(layers_param[8]);
	layers[9] = new DropoutLayer<float>(layers_param[9]);
	layers[10] = new InnerProductLayer<float>(layers_param[10]);
	layers[11] = new ReluLayer<float>(layers_param[11]);
	layers[12] = new DropoutLayer<float>(layers_param[12]);
	layers[13] = new InnerProductLayer<float>(layers_param[13]);
	layers[14] = new Logistic<float>(layers_param[14]);

	int j = 0;
	for (int i = 0; i < model_component->getNumLayers(); ++i) {
		voc_model->setLayers(layers[i]);
		if (layers_param[i]->getParamTrainType() == NEED) {
			voc_model->setNeedTrainLayers(layers[i]);
			j++;
		}
	}
	
	cout << "Initialize layers is done.\n";
	Matrix<float> *w[model_component->getNumNeedTrainLayers()];
	Matrix<float> *bias[model_component->getNumNeedTrainLayers()];

	vector<Layer*> train_layers = model_component->getNeedTrainLayers();

	for (int j = 0; j < model_component->getNumNeedTrainLayers(); ++j) {
		w[j] = train_layers[j]->getW();
		bias[j] = train_layers[j]->getBias();
	}

	for (int k = 0; k < model_component->getNumNeedTrainLayers(); ++k) {
		MPI_Bcast(w[k]->getDevData(), model_component->getWLen()[k], \
		           MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(bias[k]->getDevData(), model_component->getBiasLen()[k], \
		           MPI_FLOAT, 0, MPI_COMM_WORLD);
	}

	num_train_per_process = model_component->getNumTrain() / (num_process - 1);
	num_valid_per_process = model_component->getNumvalid() / (num_process - 1);

	int mini_data_len = layers_param->getMinibatchSize() * model_component->getInLen();
	int mini_label_len = layers_param->getMinibatchSize();

	int train_data_len_part = num_train_per_process * model_component->getInLen();
	int train_label_len_part = num_train_per_process;
	int valid_data_len_part = num_valid_per_process * model_component->getInLen();
	int valid_label_len_part = num_valid_per_process;

	Matrix<float>* train_data = new Matrix<float>(num_train_per_process, \
								model_component->getInLen());
	Matrix<float>* train_label = new Matrix<float>(num_train_per_process, 1);
	Matrix<float>* valid_data = new Matrix<float>(num_valid_per_process, \
								model_component->getInLen());
	Matrix<float>* valid_label = new Matrix<float>(num_valid_per_process, 1);

	Matrix<float>* mini_data = new Matrix<float>(layers_param->getMinibatchSize(), \
									model_component->getInLen());
	Matrix<float>* mini_label = new Matrix<float>(layers_param->getMinibatchSize(), 1);

	MPI_Status status;
	MPI_Recv(train_data->getDevData(), train_data_len_part, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);
	MPI_Recv(train_label->getDevData(), train_label_len_part, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);
	MPI_Recv(valid_data->getDevData(), valid_data_len_part, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);
	MPI_Recv(valid_label->getDevData(), valid_label_len_part, \
			MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status);

	cout << "Each process reciving their data...\n";



	Matrix<float> *y[model_component->getNumLayers()];
	Matrix<float> *dE_dy[model_component->getNumLayers()];
	for (int j = 0; j < model_component->getNumLayers(); ++j) {
		y[j] = layers[j]->getY();
		dE_dy[j] = layers[j]->getDEDY();
	}

	int passMsg = 0;
	const int num_pars_type = 2*model_component->getNumNeedTrainLayers();

	clock_t t;
	t = clock();
	clock_t t1;
	t1 = clock();
	Matrix<int>* total_record = new Matrix<int>(softmax_fcp->getNumIn(), softmax_fcp->getNumIn());

	cout << "Start training...\n";

	for(int epoch_idx = 0; epoch_idx < num_epoch; epoch_idx++){
		int error = 0;
		float likelihood = 0;
		softmax.setRecordToZero();	

		for(int batch_idx = 0; batch_idx < num_minibatch; batch_idx++){

			mini_data->changePtrFromStart(train_data->getDevData(), \
					mini_data_len * batch_idx);
			mini_label->changePtrFromStart(train_label->getDevData(), \
					mini_label_len * batch_idx);

			layers[0].computeOutputs(mini_data);
			for (int k = 1; k < model_component->getNumLayers(); ++k) {
				layers[k].computeOutputs(y[k-1]);
			}

			likelihood += layers[model_component->getNumLayers()-1].computeError(mini_label, error);
			layers[model_component->getNumLayers()-1].computeDerivsOfInput(inner2_dE_dy, mini_label);

			for (int k = model_component->getNumLayers()-2; \
						k >= 1; --k) {
				layers[k].computeDerivsOfInput(dE_dy[k-1]);
			}

			for (int k = model_component->getNumNeedTrainLayers()-1; \
						k >= 1; --k) {
				train_layers[k].computeDerivsOfPars(y[k-1]);
			}


			inner2.computeDerivsOfPars(relu4_y);

			inner1.computeDerivsOfPars(pool3_y);

			cnn3.computeDerivsOfPars(cnn3_y);

			cnn2.computeDerivsOfPars(cnn2_y);

			cnn1.computeDerivsOfPars(mini_data);

			cnn1.updatePars();
			cnn2.updatePars();
			cnn3.updatePars();
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

					#pragma omp parallel num_threads(2)
					{
						int t_tid = omp_get_thread_num();
						if (t_tid == 0) {
							MPI_Send(w[pars_addr], model_component->getWLen()[pars_addr], \
				               MPI_FLOAT, 0, swap_id+passMsg, \
						       MPI_COMM_WORLD);
						}else {
							MPI_Send(bias[pars_addr], model_component->getBiasLen()[pars_addr], \
				               MPI_FLOAT, 0, swap_id+passMsg, \
						       MPI_COMM_WORLD);
						}
					}
					
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
					#pragma omp parallel num_threads(2)
					{
						int t_tid = omp_get_thread_num();
						if (t_tid == 0) {
							MPI_Recv(w[pars_addr], model_component->getWLen()[pars_addr], \
                               MPI_FLOAT, 0, swap_id + passMsg, \
                               MPI_COMM_WORLD, &status);
						} else {
							MPI_Recv(bias[pars_addr], model_component->getBiasLen()[pars_addr], \
                               MPI_FLOAT, 0, swap_id + passMsg, \
                               MPI_COMM_WORLD, &status);
						}
					}
				}
			}
	
			if(batch_idx == num_minibatch - 1){
				
				softmax.setRecordToZero();	
				int errorValid = 0;
				float loglihoodValid = 0;
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

					cnn3.computeOutputs(pool2_y);
					relu3.computeOutputs(cnn3_y);
					pool3.computeOutputs(relu3_y);

					inner1.computeOutputs(pool3_y);
					relu4.computeOutputs(inner1_y);
					inner2.computeOutputs(relu4_y);
					softmax.computeOutputs(inner2_y);

					loglihoodValid += softmax.computeError(mini_label, errorValid);

				}
				Matrix<int>* process_record = softmax.getResultRecord();
				int totalValid = errorValid;
				total_record->copyFromDevice(process_record);
				//0号进程不能参与传递参数，故没有使用reduce
				if(num_process > 2){
					if(rank == 1){
						for(int i = 2; i < num_process; i++){
							MPI_Recv(&errorValid, 1, MPI_INT, i, i, \
									MPI_COMM_WORLD, &status);   
							totalValid += errorValid;
							MPI_Recv(process_record->getDevData(), \
									process_record->getNumEles(), \
									MPI_INT, i, i*10, MPI_COMM_WORLD, &status);
							total_record->add(process_record, 1, 1);
						}       
					}else{  
						MPI_Send(&errorValid, 1, MPI_INT, 1, rank, MPI_COMM_WORLD);
						MPI_Send(process_record->getDevData(), \
								process_record->getNumEles(), \
								MPI_INT, 1, rank * 10, MPI_COMM_WORLD);
					}       
				}       

				if(rank == 1){
					cout << "train: epoch_idx: " << epoch_idx << ", accuracy: " \
						<<  1 - (float)error/num_train_per_process  \
						<< ",likelihood: "<< likelihood<< endl;
					cout << "      valid: epoch_idx: " << epoch_idx << ", accuracy: " \
						<<  1 - (float)totalValid/num_valid_per_process \
						<< ",likelihood: "<< loglihoodValid<< endl;
					total_record->showValue("Confusion matrix");
				}
			}
		}
		
		if(rank == 1){
			t1 = clock() - t1;
			cout << " " << ((float)t1/CLOCKS_PER_SEC) << " seconds.\n";
			t1 = clock();

			cnn1_w->showValue("cnn1_w");
			cnn1_bias->showValue("cnn1_bias");
			cnn2_w->showValue("cnn2_w");
			cnn2_bias->showValue("cnn2_bias");
			cnn3_w->showValue("cnn3_w");
			cnn3_bias->showValue("cnn3_bias");
			inner1_w->showValue("inner1_w");
			inner1_bias->showValue("inner1_bias");
			softmax_w->showValue("inner2_w");
			softmax_bias->showValue("inner2_bias");
		}

	}
	if(rank == 1){
		t = clock() - t;
		cout << " " << ((float)t/CLOCKS_PER_SEC) / num_epoch << " seconds.\n";
		t = clock();
	}

	delete mini_data;
	delete mini_label;
	delete train_data;
	delete train_label;
	delete valid_data;
	delete valid_label;
	delete total_record;
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
	int n_push = 49;
	int n_fetch = 50;

	int conv1_in_size = 32;
	int conv1_in_channel = 3;
	int conv1_pad = 2;
	int conv1_stride = 1;
	int conv1_filter_size = 5;
	int conv1_out_channel = 8;
	float conv1_w_lr = 0.001;
	float conv1_b_lr = 0.001;
	float conv1_momentum = 0.9;
	float conv1_weight_decay = 0;


	//sigmoid层参数可以先忽略
	int pool1_pad = 0;
	int pool1_stride = 2;
	int pool1_filter_size = 3;
	PoolingType pool1_type = MAX_POOLING;

	int conv2_pad = 2;
	int conv2_stride = 1;
	int conv2_filter_size = 5;
	int conv2_out_channel = 16;
	float conv2_w_lr = 0.001;
	float conv2_b_lr = 0.001;
	float conv2_momentum = 0.9;
	float conv2_weight_decay = 0;

	int pool2_pad = 0;
	int pool2_stride = 2;
	int pool2_filter_size = 3;
	PoolingType pool2_type = AVG_POOLING;

	int conv3_pad = 2;
	int conv3_stride = 1;
	int conv3_filter_size = 5;
	int conv3_out_channel = 32;
	float conv3_w_lr = 0.001;
	float conv3_b_lr = 0.001;
	float conv3_momentum = 0.9;
	float conv3_weight_decay = 0;

	int pool3_pad = 0;
	int pool3_stride = 2;
	int pool3_filter_size = 3;
	PoolingType pool3_type = MAX_POOLING;

	int inner1_num_out = 64;
	float inner1_w_lr = 0.001;
	float inner1_b_lr = 0.001;
	float inner1_momentum = 0.9;
	float inner1_weight_decay = 0;

	int inner2_num_out = 10;
	float inner2_w_lr = 0.001;
	float inner2_b_lr = 0.001;
	float inner2_momentum = 0.9;
	float inner2_weight_decay = 0;

	int num_layers = 15;
	Param* params[num_layers];

	params[0] = new ConvParam("conv1_layer", minibatch_size, \
			conv1_w_lr, conv1_b_lr, conv1_momentum, conv1_weight_decay, \
			n_push, n_fetch, conv1_in_size, conv1_pad, conv1_stride, \
			conv1_in_channel, conv1_filter_size, conv1_out_channel);

	params[1] = new FullConnectParam("relu1_layer", \
			0, conv1_cp);

	params[2] = new PoolParam("pool1_layer", pool1_pad, \
			pool1_stride, pool1_filter_size, 0, conv1_cp, pool1_type);

	params[3] = new ConvParam("conv2_layer", conv2_w_lr, \
			conv2_b_lr, conv2_momentum, conv2_weight_decay, n_push, \
			n_fetch, conv2_pad, conv2_stride, conv2_filter_size, \
			conv2_out_channel, pool1_pp);

	params[4] = new FullConnectParam("relu2_layer", \
			0, conv2_cp);

	params[5] = new PoolParam("pool2_layer", pool2_pad, \
			pool2_stride, pool2_filter_size, 0, conv2_cp, pool2_type);

	params[6] = new ConvParam("conv3_layer", conv3_w_lr, \
			conv3_b_lr, conv3_momentum, conv3_weight_decay, n_push, \
			n_fetch, conv3_pad, conv3_stride, conv3_filter_size, \
			conv3_out_channel, pool2_pp);

	params[7] = new FullConnectParam("relu3_layer", \
			0, conv3_cp);

	params[8] = new PoolParam("pool3_layer", pool3_pad, \
			pool3_stride, pool3_filter_size, 0, conv3_cp, pool3_type);

	params[9] = new FullConnectParam("drop1_layer", \
			0, pool3_pp);

	params[10] = new InnerParam("inner1_layer", inner1_w_lr, \
			inner1_b_lr, inner1_momentum, inner1_weight_decay, n_push, \
			n_fetch, inner1_num_out, pool3_pp);

	params[11] = new FullConnectParam("relu4_layer", \
			0, inner1_ip);

	params[12] = new FullConnectParam("drop2_layer", \
			0, inner1_ip);

	params[13] = new InnerParam("inner2_layer", inner2_w_lr, \
			inner2_b_lr, inner2_momentum, inner2_weight_decay, \
			n_push, n_fetch, inner2_num_out, relu4_y);

	params[14] = new FullConnectParam("softmax_layer", \
			0, inner2_ip);

	params[0]->setMinibatchSize(minibatch_size);
	params[0]->setNPush(n_push);
	params[0]->setNFetch(n_fetch);

	int in_len = params[0]->getInSize() * params[0]->getInSize() \
 				* params[0]->getInChannel();

	ModelComponent* voc_model = new ModelComponent;
	voc_model->setInLen(in_len);
	voc_model->setNumLayers(num_layers);

	int j = 0;
	for (int i = 0; i < num_layers; ++i) {
		voc_model->setLayersParam(params[i]);
		if (params[i]->getParamTrainType() == NEED) {
			voc_model->setNeedTrainLayersParam(params[i]);
			j++;
		}
	}

	int num_need_train_layers = 5;
	int num_need_train_local_layers = 3;
	voc_model->setNumNeedTrainLayers(num_need_train_layers);
	voc_model->setNumNeedTrainLocalLayers(num_need_train_local_layers);
	vector<Param*> train_params = voc_model->getNeedTrainLayersParam();
	int *w_len = new int[num_need_train_layers];
	int *bias_len = new int[num_need_train_layers];

	for (int k = 0; k < num_need_train_local_layers; ++k) {
		w_len[k] = train_params[k]->getOutChannel() * train_params[k]->getFilterSize() \
			* train_params[k]->getFilterSize() * train_params[k]->getInChannel();
		bias_len[k] = train_params[k]->getOutChannel();
		voc_model->setWLen(w_len[k]);
		voc_model->setBiasLen(bias_len[k]);
	}
	for (int k = num_need_train_local_layers; k < num_need_train_layers; ++k) {
		w_len[k] = train_params[k]->getNumIn() * train_params[k]->getNumOut();
		bias_len[k] = train_params[k]->getNumOut();
		voc_model->setWLen(w_len[k]);
		voc_model->setBiasLen(bias_len[k]);
	}

	if(rank == 0){ 
		managerNode(voc_model);
	}   
	else{
		workerNode(voc_model);
	} 	

	delete voc_model;

	MPI_Finalize();
	return 0;
}



















