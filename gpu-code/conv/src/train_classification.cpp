///
/// \file train_classification.cpp
/// @brief


#include <iostream>
#include <algorithm>
#include <sstream>
#include "train_classification.hpp"

using namespace std;

template <typename Dtype>
void TrainClassification<Dtype>::createPixelAndLabel(){
	int times = 0;
	if (this->_model_component->_pid == 0) 
		times = (this->_model_component->_num_process - 1);
	else
		times = 1;
	for(int i = 0; i < times*2; i++){
		Matrix<Dtype> *pixel = new Matrix<Dtype>(this->_model_component->_minibatch_size, \
				this->_model_component->_one_img_len);
		Matrix<int> *label = new Matrix<int>(this->_model_component->_minibatch_size, 1);
		this->_model_component->_mini_data.push_back(pixel);
		this->_model_component->_mini_label.push_back(label);
	}
}

template <typename Dtype>
void TrainClassification<Dtype>::parseImgBinary(int num_process){
	this->_model_component->_num_process = num_process;

	if(this->_model_component->_pid == this->_model_component->_master_pid){
		this->_load_layer = new LoadDIC<Dtype>(this->_model_component->_minibatch_size);
	}
	TrainModel<Dtype>::parseImgBinary(num_process);
}

template <typename Dtype>
void TrainClassification<Dtype>::forwardLastLayer(){

	this->_model_component->_layers[this->_model_component->_num_layers-1]->computeOutput(\
				this->_model_component->_y_for_worker[this->_model_component->_num_layers-1]);
	this->_likelihood += dynamic_cast<Logistic<Dtype>* >( \
			this->_model_component->_layers[this->_model_component->_num_layers-1]) \
				   ->computeError(this->_model_component->_mini_label_for_compute, this->_error);
}

template <typename Dtype>
void TrainClassification<Dtype>::backwardLastLayer(){
	Logistic<Dtype> *last_layer = dynamic_cast<Logistic<Dtype>* >( \
			this->_model_component->_layers[this->_model_component->_num_layers-1]);
	last_layer->computeDerivsOfInput(this->_model_component->_dE_dy_for_worker[ \
			this->_model_component->_num_layers-2], \
			this->_model_component->_mini_label_for_compute);
}

template <typename Dtype>
void TrainClassification<Dtype>::createMPIDist() {

	TrainModel<Dtype>::createMPIDist();

	int num_trans;
	if(this->_model_component->_pid == 0)
		num_trans = this->_model_component->_num_process - 1;
	else
		num_trans = 1;

	//train和valid，然后是下一个进程的
	MPIDistribute<int> *send_label[2*num_trans];
	int label_len = this->_model_component->_minibatch_size;

	for(int i = 0; i < num_trans; i++){
		int trans_pid = 0;
		if(this->_model_component->_pid == 0)
			trans_pid = i+1;

		send_label[i*2] = new MPIDistribute<int>( \
					label_len, i+num_trans, trans_pid, MPI_INT, \
					this->_model_component->_mini_label[i*2]->getDevData());
		send_label[i*2+1] = new MPIDistribute<int>( \
					label_len, i+num_trans*3, trans_pid, MPI_INT, \
					this->_model_component->_mini_label[i*2+1]->getDevData());

		this->_model_component->_send_recv_label.push_back(send_label[i*2]);
		this->_model_component->_send_recv_label.push_back(send_label[i*2+1]);
	}
}


template <typename Dtype>
void TrainClassification<Dtype>::train() {

	int flag = 0;
	clock_t t;
	t = clock();
	for (int epoch_idx = 0; epoch_idx < this->_model_component->_num_epoch; \
			epoch_idx++) {
		this->_model_component->_y_for_worker[0] = this->_model_component->_mini_data[0];
		this->_model_component->_mini_label_for_compute= this->_model_component->_mini_label[0];

		this->_likelihood = 0;
		this->_error = 0;

		Logistic<Dtype> *last_layer = dynamic_cast<Logistic<Dtype>* >( \
					this->_model_component->_layers[this->_model_component->_num_layers-1]);
		last_layer->setRecordToZero();


		for(int batch_idx = 0; batch_idx < this->_model_component->_num_train_batch; \
				batch_idx++){
			if((epoch_idx == this->_model_component->_num_epoch - 1 \
					&& batch_idx == this->_model_component->_num_train_batch - 1) \
					|| this->_is_stop == true)
				flag = PROCESS_END;
			else
				flag = batch_idx;

			this->_model_component->_send_recv_pixel[0]->sendFlag(flag);
			this->_model_component->_send_recv_label[0]->setFlag(flag);
			this->_model_component->_send_recv_pixel[0]->dataFrom();
			this->_model_component->_send_recv_label[0]->dataFrom();

			/*
			if(batch_idx == this->_model_component->_num_train_batch-1){
				this->_model_component->_mini_data[0]->savePars("snapshot/input_snap/mini_data.bin");
				this->_model_component->_mini_label[0]->savePars("snapshot/input_snap/mini_label.bin");
			}*/

			this->forwardPropagate();
			forwardLastLayer();
			backwardLastLayer();
			this->backwardPropagate();
/*
	t = clock() - t;
	cout << " backward: "<< ((float)t/CLOCKS_PER_SEC) << "s.\n";
	t = clock();
	cout << batch_idx << ": update\n";
*/
			this->computeAndUpdatePars();
/*
	t = clock() - t;
	cout << " update: "<< ((float)t/CLOCKS_PER_SEC) << "s.\n";
	t = clock();
*/	
			this->sendAndRecvWBiasForWorker(epoch_idx, batch_idx, flag);

			if(batch_idx == this->_model_component->_num_train_batch-1){
				cout << "----------epoch_idx: " << epoch_idx << "-----------\n";
				cout << "training likelihood: " << this->_likelihood << endl;
				cout << "classification training accuarcy: " << 1-(float)this->_error/ \
					(this->_model_component->_num_train_batch \
					 *this->_model_component->getMinibatchSize()) << endl;
				Matrix<int>* train_record = last_layer->getResultRecord();
				train_record->showValue("train record");

				this->_likelihood = 0;
				this->_error = 0;

				this->_model_component->_y_for_worker[0] = this->_model_component->_mini_data[1];
				this->_model_component->_mini_label_for_compute \
						= this->_model_component->_mini_label[1];
				
				last_layer->setRecordToZero();
				
				for(int valid_idx = 0; \
						valid_idx < this->_model_component->_num_valid_batch; \
						valid_idx++){
					if(epoch_idx == this->_model_component->_num_epoch - 1 \
							&& valid_idx == this->_model_component->_num_valid_batch - 1 \
							|| this->_is_stop == true)
						flag = PROCESS_END;
					else
						flag = valid_idx;

					this->_model_component->_send_recv_pixel[1]->sendFlag(flag);
					this->_model_component->_send_recv_label[1]->setFlag(flag);
					this->_model_component->_send_recv_pixel[1]->dataFrom();
					this->_model_component->_send_recv_label[1]->dataFrom();

					this->_model_component->_mini_data[1]->savePars("snapshot/input_snap/mini_data.bin");
					this->_model_component->_mini_label[1]->savePars("snapshot/input_snap/mini_label.bin");
					this->forwardPropagate();
					forwardLastLayer();

				}
				Matrix<int>* valid_record = last_layer->getResultRecord();
				valid_record->showValue("valid record");

				cout << "validation likelihood: " << this->_likelihood << endl;
				cout << "classification valid accuarcy: " << 1-(float)this->_error/ \
					(this->_model_component->_num_valid_batch \
					 *this->_model_component->getMinibatchSize()) << endl;

		
			}
		}

	//	for(int i = 0; i < this->_model_component->_num_need_train_layers; i++){
	//			this->_model_component->_w[i]->showValue( \
	//					this->_model_component->_layers_need_train_param[i]->getName()+"_w");
	//	}

//		if(this->_is_stop == false)
//			earlyStopping(epoch_idx);

/*		if((epoch_idx+1) % 5 == 0 ){
			for(int i = 0; i < this->_model_component->_num_need_train_layers; i++){
				dynamic_cast<TrainParam*>( \
					this->_model_component->_layers_need_train_param[i])->lrMultiScale(0.8);
			}
		}
*/
		if(this->_model_component->_pid == 1){
			t = clock() - t;
			cout << ((float)t/CLOCKS_PER_SEC) << "s.\n";
			t = clock();
		}
		
	}
}

template <typename Dtype>
void TrainClassification<Dtype>::sendAndRecvForManager() {
	int num_pars_type = this->_model_component->_num_need_train_layers;
	int	num_trans = this->_model_component->_num_process - 1;

	int pixel_len = this->_model_component->_minibatch_size*this->_model_component->_one_img_len;
	int label_len = this->_model_component->_minibatch_size;

	int num_threads = num_trans*num_pars_type+num_trans*2;
	cout << "num_threads: " << num_threads<< endl;
#pragma omp parallel num_threads(num_threads)
	{
		int tid = omp_get_thread_num();
	  	Dtype *h_mini_pixel = new Dtype[pixel_len];   //分配在主机内存上
	   	int *h_mini_label = new int[label_len]; 
		if(tid < num_trans*2){
			int pid = tid / 2 + 1 + this->_model_component->_master_pid;   //计算出对应的进程ID
			int type_id = tid % 2;   //计算是train还是valid
			do{
				if(type_id == 0)
					this->_load_layer->loadTrainOneBatch(this->_model_component->_send_recv_pixel[tid]->getFlag()+1, \
								num_trans, pid-1, h_mini_pixel, h_mini_label);
				else
					this->_load_layer->loadValidOneBatch(this->_model_component->_send_recv_pixel[tid]->getFlag()+1, \
								num_trans, pid-1, h_mini_pixel, h_mini_label);
				
				this->_model_component->_mini_data[tid]->copyFromHost(h_mini_pixel, pixel_len);
				this->_model_component->_mini_label[tid]->copyFromHost(h_mini_label, label_len);

				this->_model_component->_send_recv_pixel[tid]->receviceFlag();
				this->_model_component->_send_recv_label[tid]->setFlag( \
						this->_model_component->_send_recv_pixel[tid]->getFlag());
				this->_model_component->_send_recv_pixel[tid]->dataTo();
				this->_model_component->_send_recv_label[tid]->dataTo();

				if(type_id == 0){
				   if(this->_model_component->_send_recv_pixel[tid]->getFlag() \
						== this->_model_component->_num_train_batch-1){
					this->_model_component->_send_recv_pixel[tid]->setFlag(-1);
					this->_model_component->_send_recv_label[tid]->setFlag(-1);
					}
				}else{
					if(this->_model_component->_send_recv_pixel[tid]->getFlag() \
						== this->_model_component->_num_valid_batch-1){
						this->_model_component->_send_recv_pixel[tid]->setFlag(-1);
						this->_model_component->_send_recv_label[tid]->setFlag(-1);
					}
				}
			}while(this->_model_component->_send_recv_pixel[tid]->getFlag() != PROCESS_END);

		}else{
			tid -= 2*num_trans;

			do{
				this->_model_component->_send_recv_w[tid]->receviceFlag();
				this->_model_component->_send_recv_bias[tid]->setFlag(this->_model_component->_send_recv_w[tid]->getFlag());
				//偶数是子进程向0号请求，奇数是子进程发送给0号
				if(this->_model_component->_send_recv_w[tid]->getFlag() % 2 == 0){
					this->_model_component->_send_recv_w[tid]->dataTo();
					this->_model_component->_send_recv_bias[tid]->dataTo();
				}else{
					this->_model_component->_send_recv_w[tid]->dataFrom();
					this->_model_component->_send_recv_bias[tid]->dataFrom();
				}
			}while(this->_model_component->_send_recv_w[tid]->getFlag() != PROCESS_END);
		}
	}
}	


