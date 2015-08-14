///
///  \file conv3.cu
///

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <omp.h>
#include "mpi.h"
#include "train_model.hpp"

using namespace std;

#define THREAD_END 100000
	
typedef void(*loadFun)(); 

int Param::_minibatch_size;

void managerNode(TrainModel *model){

	cout << "Loading data...\n";

	cout << "Loading data is done.\n";

	cout << "Initialize weight and bias...\n";
/*
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
	*/
}


void workerNode(ModelComponent* model_component){

	cout << "Initialize layers...\n";

	cout << "Initialize layers is done.\n";

	cout << "Each process reciving their data...\n";
/*
	int passMsg = 0;
	const int num_pars_type = 2*model_component->getNumNeedTrainLayers();

	clock_t t;
	t = clock();
	clock_t t1;
	t1 = clock();
	cout << "Start training...\n";

	for(int epoch_idx = 0; epoch_idx < num_epoch; epoch_idx++){

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

	}
	*/

}

int main(int argc, char** argv){

	int pid; 
	int num_process;
	int prov;
	MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE, &prov);
	if (prov < MPI_THREAD_MULTIPLE)
	{   
		printf("Error: the MPI library doesn't provide the required thread level\n");
		MPI_Abort(MPI_COMM_WORLD, 0); 
	}   
	MPI_Comm_pid(MPI_COMM_WORLD,&pid);
	MPI_Comm_size(MPI_COMM_WORLD,&num_process);

	if(num_process <= 1){
		printf("Error: process number must bigger than 1\n");
		MPI_Abort(MPI_COMM_WORLD, 0); 
	}

	//检测有几个gpu
	int num_gpu;
	cudaGetDeviceCount(&num_gpu);
	cudaSetDevice(pid % num_gpu);

	cout << num_gpu << endl;
	cout << num_process << endl;	

	TrainModel *voc_model = new TrainModel();

	if(pid == 0){ 
		managerNode(voc_model);
	}   
	else{
		workerNode(voc_model);
	} 	

	delete voc_model;
	MPI_Finalize();
	return 0;
}

















