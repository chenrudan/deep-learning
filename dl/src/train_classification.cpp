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
	this->_model_component->_mini_data = new Matrix<Dtype>(this->_model_component->_minibatch_size, \
				this->_model_component->_one_img_len);
	this->_model_component->_mini_label	= new Matrix<int>(this->_model_component->_minibatch_size, 1);
}

template <typename Dtype>
void TrainClassification<Dtype>::parseImgBinary(string train_file, string valid_file){
	this->_load_layer = new LoadCifar10<Dtype>(this->_model_component->_minibatch_size);
	this->_model_component->_num_train = this->_load_layer->getNumTrain();
	this->_model_component->_num_valid = this->_load_layer->getNumValid();
	this->_model_component->setNumTrainBatch();
	this->_model_component->setNumValidBatch();

}

template <typename Dtype>
void TrainClassification<Dtype>::forwardLastLayer(){

	this->_model_component->_layers[this->_model_component->_num_layers-1]->computeOutput(\
			this->_model_component->_y[this->_model_component->_num_layers-1]);
	this->_likelihood += dynamic_cast<Logistic<Dtype>* >( \
			this->_model_component->_layers[this->_model_component->_num_layers-1]) \
						 ->computeError(this->_model_component->_mini_label, this->_error);
}

template <typename Dtype>
void TrainClassification<Dtype>::backwardLastLayer(){
	Logistic<Dtype> *last_layer = dynamic_cast<Logistic<Dtype>* >( \
			this->_model_component->_layers[this->_model_component->_num_layers-1]);
	last_layer->computeDerivsOfInput(this->_model_component->_dE_dy[ \
			this->_model_component->_num_layers-2], \
			this->_model_component->_mini_label);
}

template <typename Dtype>
void TrainClassification<Dtype>::train() {

	clock_t t;
	t = clock();

	int pixel_len = this->_model_component->_minibatch_size*this->_model_component->_one_img_len;
	int label_len = this->_model_component->_minibatch_size;
	Dtype *h_mini_pixel = new Dtype[pixel_len];   //分配在主机内存上
	int *h_mini_label = new int[label_len]; 

	for (int epoch_idx = 0; epoch_idx < this->_model_component->_num_epoch; \
			epoch_idx++) {

		this->_likelihood = 0;
		this->_error = 0;

		Logistic<Dtype> *last_layer = dynamic_cast<Logistic<Dtype>* >( \
				this->_model_component->_layers[this->_model_component->_num_layers-1]);
		last_layer->setRecordToZero();


		for(int batch_idx = 0; batch_idx < this->_model_component->_num_train_batch; \
				batch_idx++){

			this->_load_layer->loadTrainOneBatch(batch_idx, h_mini_pixel, h_mini_label);
			this->_model_component->_mini_data->copyFromHost(h_mini_pixel, \
						pixel_len);
			this->_model_component->_mini_label->copyFromHost(h_mini_label, \
						label_len);
			this->forwardPropagate();
			forwardLastLayer();
			backwardLastLayer();
			this->backwardPropagate();
			
			this->computeAndUpdatePars();

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

				last_layer->setRecordToZero();

				for(int valid_idx = 0; \
						valid_idx < this->_model_component->_num_valid_batch; \
						valid_idx++){
						
					this->_load_layer->loadValidOneBatch( valid_idx, \
						h_mini_pixel, h_mini_label);
					this->_model_component->_mini_data->copyFromHost(h_mini_pixel, \
						pixel_len);
					this->_model_component->_mini_label->copyFromHost(h_mini_label, \
						label_len);

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

		t = clock() - t;
		cout << ((float)t/CLOCKS_PER_SEC) << "s.\n";
		t = clock();

	}
}
