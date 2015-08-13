/*
 * filename:conv3.cu
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <omp.h>
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
	
typedef void(*loadFun)(); 

int num_process;
int rank;

int num_epoch = 500;
int Param::_minibatch_size = 100;
int TrainParam::_n_push = 49;
int TrainParam::_n_fetch = 50;

void managerNode(ModelComponent *model_component){

	cout << "Loading data...\n";

	//	savePars(valid_data, "./snapshot/input_snap/valid_data.bin");

	cout << "Loading data is done.\n";

	cout << "Initialize weight and bias...\n";

	//一个线程负责图片数据的交互，一个线程负责权重的交互
	#pragma omp parallel num_threads(2 * (num_process - 1)) 
	{
		int num_pars_type;  ///>需要传输的数据种类
		int trans_ops;   ///>需要传输的数据种类以及方式
		
		int t_tid = omp_get_thread_num();
		if(tid == 0){	
			num_pars_type = 4;
			trans_ops = 2;  ///>控制线程只需要发送给执行线程，下面是因为既要发送又要接收

			float* my_pars[num_pars_type] = {train_data->getDevData(), train_label->getDevData, \
					valid_data->getDevData(), valid_label->getDevData()};
			int pars_len[num_pars_type] = {train_data_len_part, train_label_len_part, \
					valid_data_len_part, valid_label_len_part};
			loadFun pLoadFun[2] = {&voc.loadTrainOneBatch, &voc.loadValidOneBatch};

			#pragma omp parallel num_threads(trans_ops * (num_process - 1)) 
			{
				MPI_Status status;
				int par_state = 0;

				int tid = omp_get_thread_num();   
				int pid = tid / trans_ops + 1;  ///>计算需要把数据传递给哪个进程
				int swap_id = tid % trans_ops;  ///>计算这个线程具体处理的是train还是valid
					
				while(par_state != THREAD_END){
					pLoadFun[swap_id]();
					MPI_Send(my_pars[swap_id*trans_op]+(pid-1)*train_data_len_part, \
							train_data_len_part, MPI_FLOAT, pid, pid, MPI_COMM_WORLD);
					MPI_Send(my_pars[swap_id*trans_op+1]+(pid-1)*train_label_len_part, \
							train_label_len_part, MPI_VECTOR, pid, pid, MPI_COMM_WORLD);
				}
			}
			
		}else{
			num_pars_type = 2*model_component->getNumNeedTrainLayers();
			trans_ops = 2*num_pars_type;

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
	}
	for (int k = 0; k < model_component->getNumNeedTrainLayers(); ++k) {
		delete w[k];
		delete bias[k];
	}

	delete train_data;
	delete train_label;
	delete valid_data;
	delete valid_label;
}


void workerNode(ModelComponent* model_component){

	cout << "Initialize layers...\n";

	cout << "Initialize layers is done.\n";

	cout << "Each process reciving their data...\n";

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

		for(int batch_idx = 0; batch_idx < model_component->getNumTrainBatch(); batch_idx++){

			mini_data->changePtrFromStart(train_data->getDevData(), \
					mini_data_len * batch_idx);
			mini_label->changePtrFromStart(train_label->getDevData(), \
					mini_label_len * batch_idx);
			/*
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
				train_layers[k].computeDerivsOfPars(y_for_compute_y[k-1]);
			}


			 */
			if((batch_idx + 1) % conv1_cp->getNPush() == 0){
				if(epoch_idx == num_epoch - 1){
					if((batch_idx + conv1_cp->getNPush()) >= model_component->getNumTrainBatch() \
							|| batch_idx == model_component->getNumTrainBatch() - 1)
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
					if((batch_idx + conv1_cp->getNFetch()) >= model_component->getNumTrainBatch() \
							|| batch_idx == model_component->getNumTrainBatch() - 1)
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

			//cout << batch_idx << "??\n";
			/*	
				if(batch_idx == model_component->getNumTrainBatch() - 1){



				softmax.setRecordToZero();	
				int errorValid = 0;
				float loglihoodValid = 0;
				for(int validIdx = 0; validIdx < model_component->getNumValidBatch(); validIdx++){

				mini_data->changePtrFromStart(valid_data->getDevData(), \
				mini_data_len * validIdx);
				mini_label->changePtrFromStart(valid_label->getDevData(), \
				mini_label_len * validIdx);

			layers[0].computeOutputs(mini_data);
			for (int k = 1; k < model_component->getNumLayers(); ++k) {
				layers[k].computeOutputs(y[k-1]);
			}

			likelihood += layers[model_component->getNumLayers()-1].computeError(mini_label, error);

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
			<<  1 - (float)error/model_component->getNumTrainEachProcess()  \
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

		}

		*/
			//	if(epoch_idx == 10){

			//			conv1_cp->lrMultiScale(0.1);
			//			conv2_cp->lrMultiScale(0.1);
			//			conv3_cp->lrMultiScale(0.1);
			//			inner1_ip->lrMultiScale(0.1);
			//			inner2_ip->lrMultiScale(0.1);
			//		}

	}
	if(rank == 1){
		t = clock() - t;
		cout << " " << ((float)t/CLOCKS_PER_SEC) / num_epoch << " seconds.\n";
		t = clock();
	}

	delete mini_data;
	delete mini_label;
	delete total_record;
}

int main(int argc, char** argv){

	MPI_Datatype MPI_VECTOR;  ///>用来保存label和坐标
	MPI_Type_contiguous(sizeof(vector<int>)/sizeof(int), MPI_INT, &MPI_VECTOR);
	MPI_Type_commit(&MPI_VECTOR);

	int blocks[6] = {1, 1, 1, 1, 1, 1};
	MPI_Datatype types[6] = {MPI_FLOAT, MPI_VECTOR, MPI_INT, MPI_INT, MPI_INT, MPI_INT};

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

	if(rank == 0){ 
		managerNode(voc_model);
	}   
	else{
		workerNode(voc_model);
	} 	

	MPI_Finalize();
	return 0;
}

















