///
/// \file train_recommendation.cpp
/// @brief


#include <iostream>
#include <algorithm>
#include <sstream>
#include "train_recommendation.hpp"

using namespace std;

template <typename Dtype>
void TrainRecommendation<Dtype>::parseImgBinary(int num_process, \
		string train_file, string train_matches, string test_file){
	this->_model_component->_num_process = num_process;

	if(this->_model_component->_pid == this->_model_component->_master_pid){
		this->_load_layer = new LoadTianchi<Dtype>(this->_model_component->_minibatch_size, \
				num_process-1, train_file, train_matches, test_file);
	}
	TrainModel<Dtype>::parseImgBinary(num_process);
}

template <typename Dtype>
void TrainRecommendation<Dtype>::forwardLastLayer(){

	RecommendationLayer<Dtype> *last_layer = dynamic_cast<RecommendationLayer<Dtype>* >( \
			this->_model_component->_layers[this->_model_component->_num_layers-1]);
	this->_likelihood += last_layer->computeError(this->_model_component->_y_for_worker[ \
			this->_model_component->_num_layers-1], \
						   this->_model_component->_mini_label_for_compute);
	if(this->_is_test && this->_model_component->_pid != this->_model_component->_master_pid){
		Dtype *prob_record = last_layer->getProbRecord();
		int *h_labels = last_layer->getHLabel();
		for(int i = 0; i < this->_model_component->_minibatch_size/2; i++){
			fout << h_labels[2*i+1];
			fout << '\t';
			fout << h_labels[2*i];
			fout << '\t';
			fout << prob_record[i];
			fout << '\n';
		}
	}
	if(!this->_is_test){
		Dtype *prob_record = last_layer->getProbRecord();
		int *h_labels = last_layer->getHLabel();
		for(int i = 0; i < this->_model_component->_minibatch_size/2; i++){
			cout << h_labels[1+i*2] << "\t" << h_labels[2*i] << "\t";
			cout << prob_record[i] << "\n";
		}
	}
}

template <typename Dtype>
void TrainRecommendation<Dtype>::backwardLastLayer(){
	RecommendationLayer<Dtype> *last_layer = dynamic_cast<RecommendationLayer<Dtype>* >( \
			this->_model_component->_layers[this->_model_component->_num_layers-1]);
	last_layer->computeDerivsOfInput(this->_model_component->_dE_dy_for_worker[ \
			this->_model_component->_num_layers-2]);

}

template <typename Dtype>
void TrainRecommendation<Dtype>::train() {

	int flag = 0;
	clock_t t;
	t = clock();
	for (int epoch_idx = 0; epoch_idx < this->_model_component->_num_epoch; \
			epoch_idx++) {
		this->_model_component->_y_for_worker[0] = this->_model_component->_mini_data[0];
		this->_model_component->_mini_label_for_compute= this->_model_component->_mini_label[0];

		this->_likelihood = 0;

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
					
	//	this->_model_component->_mini_data[0]->showValue("data");
//			if(batch_idx == 4){
//				this->_model_component->_mini_data[0]->savePars("../snapshot/input_snap/mini_data.bin");
//				this->_model_component->_mini_label[0]->savePars("../snapshot/input_snap/mini_label.bin");
//			}
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
			
			if(batch_idx == 200 && this->_model_component->_pid == 1){
				stringstream ss;
				string str;
				ss << epoch_idx;
				ss >> str;
				for(int i = 0; i < this->_model_component->_num_need_train_layers; i++){
					this->_model_component->_w[i]->savePars( \
							"../snapshot/w_snap/"+str+"_"+this->_model_component->_layers_need_train_param[i]->getName()+"_w.bin");
					this->_model_component->_bias[i]->savePars( \
							"../snapshot/w_snap/"+str+"_"+this->_model_component->_layers_need_train_param[i]->getName()+"_bias.bin");
				}
				RecommendationLayer<Dtype> *last_layer = dynamic_cast<RecommendationLayer<Dtype>* >( \
						this->_model_component->_layers[this->_model_component->_num_layers-1]);
				last_layer->saveRecommendW("../snapshot/w_snap/"+str+"_" + "recommend_w.bin");
			}

			if(batch_idx == this->_model_component->_num_train_batch-1 && this->_model_component->_pid == 1){
				cout << "----------epoch_idx: " << epoch_idx << "-----------\n";
				cout << " training likelihood: " << this->_likelihood << endl;
			}

			if(batch_idx == this->_model_component->_num_train_batch-1 && this->_has_valid){
				this->_likelihood = 0;

				this->_model_component->_y_for_worker[0] = this->_model_component->_mini_data[1];
				this->_model_component->_mini_label_for_compute \
						= this->_model_component->_mini_label[1];
				
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

			//		this->_model_component->_mini_data[1]->savePars("../snapshot/input_snap/mini_data.bin");
			//		this->_model_component->_mini_label[1]->savePars("../snapshot/input_snap/mini_label.bin");
					this->forwardPropagate();
					forwardLastLayer();

				}
				if(this->_model_component->_pid == 1)
					cout << "validation likelihood: " << this->_likelihood << endl;
		
			}/*
			for(int i = 0; i < this->_model_component->_num_need_train_layers; i++){
				
				this->_model_component->_w[i]->showValue( \
				this->_model_component->_layers_need_train_param[i]->getName()+"_w");
			}*/
	//		this->_model_component->_y_for_worker[1]->showValue( \
				this->_model_component->_layers_param[1]->getName()+"_y");
//			this->_model_component->_y_for_worker[4]->showValue( \
				this->_model_component->_layers_param[4]->getName()+"_y");
//			this->_model_component->_y_for_worker[7]->showValue( \
				this->_model_component->_layers_param[7]->getName()+"_y");
		}
			this->_model_component->_y_for_worker[this->_model_component->_num_layers-1]->showValue( \
				this->_model_component->_layers_param[this->_model_component->_num_layers-1]->getName()+"_y");
		

//		if(this->_is_stop == false)
//			earlyStopping(epoch_idx);
/*
		if((epoch_idx+1) == 10 ){
			for(int i = 0; i < this->_model_component->_num_need_train_layers; i++){
				dynamic_cast<TrainParam*>( \
					this->_model_component->_layers_need_train_param[i])->lrMultiScale(0.1);
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
void TrainRecommendation<Dtype>::test() {

	int flag = 0;
	clock_t t;
	t = clock();
		
	RecommendationLayer<Dtype> *last_layer = dynamic_cast<RecommendationLayer<Dtype>* >( \
				this->_model_component->_layers[this->_model_component->_num_layers-1]);
	last_layer->readRecommendW("../snapshot/w_snap/8_recommend_w.bin");
	for (int epoch_idx = 0; epoch_idx < this->_model_component->_num_epoch; \
			epoch_idx++) {
		this->_model_component->_y_for_worker[0] = this->_model_component->_mini_data[0];
		this->_model_component->_mini_label_for_compute= this->_model_component->_mini_label[0];

		this->_likelihood = 0;

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
	
	//		if(batch_idx == 4){
	//			this->_model_component->_mini_data[0]->savePars("../snapshot/input_snap/mini_data.bin");
	//		}

			this->forwardPropagate();
			forwardLastLayer();
			backwardLastLayer();
			this->backwardPropagate();
		}

		cout << "----------epoch_idx: " << epoch_idx << "-----------\n";
		cout << "    testing likelihood: " << this->_likelihood << endl;

		if(this->_model_component->_pid == 1){
			t = clock() - t;
			cout << ((float)t/CLOCKS_PER_SEC) << "s.\n";
			t = clock();
		}
		
	}
}

