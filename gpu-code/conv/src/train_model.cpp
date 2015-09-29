///
/// \file train_model.cpp
/// @brief


#include <iostream>
#include <algorithm>
#include <sstream>
#include "train_model.hpp"
#include "json/json.h"
#include "inner_product_layer.hpp"
#include "logistic.hpp"
#include "sigmoid_layer.hpp"
#include "relu_layer.hpp"
#include "convnet.hpp"
#include "pooling_layer.hpp"
#include "dropout_layer.hpp"
#include "predict_object_layer.hpp"
#include "recommendation_layer.hpp"

using namespace std;

template <typename Dtype>
TrainModel<Dtype>::TrainModel(const int master_pid, const int pid, bool has_valid, bool is_test){
	_model_component = new ModelComponent<Dtype>();
	_model_component->_master_pid = master_pid;
	_model_component->_pid = pid;
	_likelihood = 0;
	_is_stop = false;
	_has_valid = has_valid;
	_is_test = is_test;
	if(has_valid)
		_num_data_type = 2;
	else
		_num_data_type = 1;
}

template <typename Dtype>
TrainModel<Dtype>::~TrainModel() {
	delete _model_component;
	delete _load_layer;
}

template <typename Dtype>
void TrainModel<Dtype>::parseImgBinary(int num_process){

	if(_model_component->_pid == _model_component->_master_pid){
		_model_component->_num_train = _load_layer->getNumTrain();
		_model_component->_num_valid = _load_layer->getNumValid();
		_model_component->_num_train_each_process = _load_layer->getNumTrain()/(num_process-1);
		_model_component->_num_valid_each_process = _load_layer->getNumValid()/(num_process-1);

		_model_component->setNumTrainBatch();
		_model_component->setNumValidBatch();

		//将batch数传递给工作进程
#pragma omp parallel num_threads(num_process-1)
		{
			int pid = omp_get_thread_num() + 1;
			MPIDistribute<int> *send_train = new MPIDistribute<int>(1, pid, \
					pid, MPI_INT, &_model_component->_num_train_batch);
			MPIDistribute<int> *send_valid = new MPIDistribute<int>(1, pid+num_process, \
					pid, MPI_INT, &_model_component->_num_valid_batch);
			send_train->dataTo();
			send_valid->dataTo();
			delete send_train;
			delete send_valid;
		}
	}else{
		MPIDistribute<int> *recv_train = new MPIDistribute<int>(1, \
				_model_component->_pid, _model_component->_master_pid, MPI_INT, \
				&_model_component->_num_train_batch);
		MPIDistribute<int> *recv_valid = new MPIDistribute<int>(1, \
				_model_component->_pid+num_process, \
				_model_component->_master_pid, MPI_INT, &_model_component->_num_valid_batch);
		recv_train->dataFrom();
		recv_valid->dataFrom();
		delete recv_train;
		delete recv_valid;

	}
}

template <typename Dtype>
void TrainModel<Dtype>::parseNetJson(string json_file) {
	Json::Reader reader;
	Json::Value root;
	ifstream fin(json_file.c_str());
	if (reader.parse(fin, root)) {
		_model_component->_minibatch_size = root["minibatch_size"].asInt();
		Param::setMinibatchSize(_model_component->_minibatch_size);
		_model_component->_n_push = root["n_push"].asInt();

		_model_component->_n_fetch = root["n_fetch"].asInt();
		_model_component->_num_epoch = root["num_epoch"].asInt();
		_model_component->_img_height = root["img_height"].asInt();
		_model_component->_img_width = root["img_width"].asInt();
		_model_component->_img_channel = root["img_channel"].asInt();

		if(_model_component->_pid == _model_component->_master_pid + 1){
			cout << "\n===========overall==============" \
				<< "\nnum_epoch: " << _model_component->_num_epoch \
				<< "\nbatchSize: " << _model_component->_minibatch_size \
				<< "\nn_fetch: " << _model_component->_n_fetch \
				<< "\nn_push: " << _model_component->_n_push;
		}

		_model_component->_num_layers = root["layer"].size();

		string layer_type, name;
		int pad_height, pad_width, stride_height, stride_width;
		int	filter_height, filter_width, filter_channel, num_out, num_in;
		float w_lr, bias_lr, momentum, weight_decay, w_gauss;
		string p_type;
		Param* param;

		for (int i = 0; i < _model_component->_num_layers; ++i) {
			layer_type = root["layer"][i]["type"].asString();
			name = root["layer"][i]["name"].asString();
			if (!root["layer"][i]["filter_height"].isNull()) {
				pad_height = root["layer"][i]["pad_height"].asInt();
				pad_width = root["layer"][i]["pad_width"].asInt();
				stride_height = root["layer"][i]["stride_height"].asInt();
				stride_width = root["layer"][i]["stride_width"].asInt();
				filter_height = root["layer"][i]["filter_height"].asInt();
				filter_width = root["layer"][i]["filter_width"].asInt();
			}
			if (!root["layer"][i]["w_lr"].isNull()) {
				w_lr = root["layer"][i]["w_lr"].asFloat();
				bias_lr = root["layer"][i]["bias_lr"].asFloat();
				momentum = root["layer"][i]["momentum"].asFloat();
				weight_decay = root["layer"][i]["weight_decay"].asFloat();
				w_gauss = root["layer"][i]["w_gauss"].asFloat();
			}
			if (!root["layer"][i]["num_out"].isNull()) {
				num_out = root["layer"][i]["num_out"].asInt();
			}
			if (!root["layer"][i]["num_in"].isNull()) {
				num_in = root["layer"][i]["num_in"].asInt();
			}
			if (!root["layer"][i]["pool_type"].isNull()) {
				p_type = root["layer"][i]["pool_type"].asString();
			}
			if (!root["layer"][i]["filter_channel"].isNull()) {
				filter_channel = root["layer"][i]["filter_channel"].asInt();
			}else{
				filter_channel = 0;
			}
			if (layer_type == "CONVOLUTION") {
				if (_model_component->_layers_param.size() == 0) {
					param = new ConvParam( \
							_model_component->_string_map_layertype[layer_type], \
							name, w_lr, bias_lr, momentum, weight_decay, w_gauss, \
							_model_component->_img_height, _model_component->_img_width, \
							pad_height, pad_width, stride_height, stride_width, \
							_model_component->_img_channel, filter_height, \
							filter_width, filter_channel);
				} else{
					param = new ConvParam( \
							_model_component->_string_map_layertype[layer_type], \
							name, w_lr, bias_lr, momentum, weight_decay, w_gauss, \
							pad_height, pad_width, stride_height, stride_width, \
							filter_height, filter_width, filter_channel, \
							dynamic_cast<LocalConnectParam*>( \
								_model_component->_layers_param.back()));
				}
			} else if (layer_type == "POOLING") {
				param = new PoolParam( \
						_model_component->_string_map_layertype[layer_type], \
						name, pad_height, pad_width, stride_height, stride_width, \
						filter_height, filter_width, 0, \
						dynamic_cast<LocalConnectParam*>( \
							_model_component->_layers_param.at( \
								_model_component->_layers_param.size() - 2)), \
						_model_component->_string_map_pooltype[p_type]);
			} else if (layer_type == "SIGMOID" || layer_type == "RECTIFIED" \
					|| layer_type == "SOFTMAX" || layer_type == "DROPOUT") {
				param = new FullConnectParam( \
						_model_component->_string_map_layertype[layer_type], \
						name, 0, _model_component->_layers_param.back());
			} else if (layer_type == "INNERPRODUCT" ) {
				if (_model_component->_layers_param.size() == 0) {
					param = new InnerParam( \
							_model_component->_string_map_layertype[layer_type], \
							name, w_lr, bias_lr, momentum, weight_decay, w_gauss, \
							num_in, num_out);
				}else{
					param = new InnerParam( \
							_model_component->_string_map_layertype[layer_type], \
							name, w_lr, bias_lr, momentum, weight_decay, w_gauss, \
							num_out, _model_component->_layers_param.back());
				}
			} else if(layer_type == "PREDICTOBJECT"){
				param = new FullConnectParam( \
						_model_component->_string_map_layertype[layer_type], \
						name, 0, _model_component->_layers_param.back());
			} else if(layer_type == "RECOMMENDSUBSTITUE"){
				param = new FullConnectParam( \
						_model_component->_string_map_layertype[layer_type], \
						name, num_out, _model_component->_layers_param.back());
			} else if(layer_type == "RECOMMENDCOMPATIBLE"){
				param = new FullConnectParam( \
						_model_component->_string_map_layertype[layer_type], \
						name, num_out, _model_component->_layers_param.back());
			}
			if(_model_component->_pid == _model_component->_master_pid + 1)
				param->printParam();
			_model_component->_layers_param.push_back(param);

			if (param->getParamTrainType() == NEED) {
				_model_component->_layers_need_train_param.push_back(param);
				_model_component->_num_need_train_layers++;
			}
		}
	}
	_model_component->_one_img_len = _model_component->_img_width \
									 *_model_component->_img_height \
									 *_model_component->_img_channel;
	_model_component->_in_len_each_process = _model_component->_one_img_len \
											 *_model_component->_minibatch_size;
}

template <typename Dtype>
void TrainModel<Dtype>::createLayerForWorker(){
	cout << _model_component->_num_layers << endl;
	for (int i = 0; i < _model_component->_num_layers; ++i){
		Layer<Dtype> *layer;
		Param *param = _model_component->_layers_param[i];
		try{
			if (param->getLayerType() == CONVOLUTION) {
				LocalConnectParam* lcp = dynamic_cast<LocalConnectParam*>(param);
				if(lcp == NULL)
					throw 5;
				layer = new ConvNet<Dtype>(dynamic_cast<ConvParam*>(lcp));
			} else if (param->getLayerType() == POOLING) {
				layer = new PoolingLayer<Dtype>(dynamic_cast<PoolParam*>(param));
			} else if (param->getLayerType() == SIGMOID) {
				layer = new SigmoidLayer<Dtype>(dynamic_cast<FullConnectParam*>(param));
			} else if (param->getLayerType() == RECTIFIED) {
				layer = new ReluLayer<Dtype>(dynamic_cast<FullConnectParam*>(param));
			} else if (param->getLayerType() == SOFTMAX) {
				layer = new Logistic<Dtype>(dynamic_cast<FullConnectParam*>(param));
			} else if (param->getLayerType() == DROPOUT) {
				layer = new DropoutLayer<Dtype>(dynamic_cast<FullConnectParam*>(param));
			} else if (param->getLayerType() == INNERPRODUCT ) {
				FullConnectParam* fcp = dynamic_cast<FullConnectParam*>(param);
				layer = new InnerProductLayer<Dtype>(dynamic_cast<InnerParam*>(fcp));
			} else if (param->getLayerType() == PREDICTOBJECT ) {
				layer = new PredictObjectLayer<Dtype>(dynamic_cast<FullConnectParam*>(param));
			} else if (param->getLayerType() == RECOMMENDSUBSTITUE ) {
				layer = new RecommendationLayer<Dtype>(dynamic_cast<FullConnectParam*>(param));
			} else if (param->getLayerType() == RECOMMENDCOMPATIBLE ) {
				layer = new RecommendationLayer<Dtype>(dynamic_cast<FullConnectParam*>(param));
			}
		}catch(int e){
			cout << "dynamic point is null\n";
		}

		layer->initCuda();
		_model_component->_layers.push_back(layer);

		if (param->getParamTrainType() == NEED) {
			_model_component->_layers_needed_train.push_back(layer);
		}
	}
}

template <typename Dtype>
void TrainModel<Dtype>::createWBiasForManager() {
	for (int i = 0; i < _model_component->_num_need_train_layers; ++i) {
		Matrix<Dtype> *w, *bias;
		int w_len, bias_len;
		Param* tp = _model_component->_layers_need_train_param[i];
		if(tp->getLayerType() == CONVOLUTION){
			ConvParam* cp = dynamic_cast<ConvParam*>(tp);
			w = new Matrix<Dtype>(cp->getFilterHeight() \
					*cp->getFilterWidth()*cp->getInChannel(), cp->getOutChannel());
			w_len = cp->getFilterHeight()*cp->getFilterWidth() \
					*cp->getInChannel()*cp->getOutChannel();
			bias = new Matrix<Dtype>(1, cp->getOutChannel());
			bias_len = cp->getOutChannel();
		}else if(tp->getLayerType() == INNERPRODUCT){
			InnerParam* ip = dynamic_cast<InnerParam*>(tp);
			w = new Matrix<Dtype>(ip->getNumIn(), ip->getNumOut());
			w_len = ip->getNumIn()*ip->getNumOut();
			bias = new Matrix<Dtype>(1, ip->getNumOut());
			bias_len = ip->getNumOut();
		}
		_model_component->_w.push_back(w);
		_model_component->_bias.push_back(bias);
		_model_component->_w_len.push_back(w_len);
		_model_component->_bias_len.push_back(bias_len);
	}
}

template <typename Dtype>
void TrainModel<Dtype>::createWBiasForWorker() {
	for (int i = 0; i < _model_component->getNumNeedTrainLayers(); ++i) {
		TrainLayer<Dtype>* tl = dynamic_cast<TrainLayer<Dtype>*>( \
				_model_component->_layers_needed_train[i]);
		_model_component->_w.push_back(tl->getW());
		_model_component->_bias.push_back(tl->getBias());
		_model_component->_w_len.push_back(tl->getW()->getNumEles());
		_model_component->_bias_len.push_back(tl->getBias()->getNumEles());
	}
}

template <typename Dtype>
void TrainModel<Dtype>::createYDEDYForWorker() {
	_model_component->_y_for_worker.push_back(_model_component->_mini_data[0]);
	_model_component->_y_needed_train.push_back(_model_component->_mini_data[0]);
	for (int i = 0; i < _model_component->_num_layers; ++i){
		_model_component->_y_for_worker.push_back( \
				_model_component->_layers[i]->getY());
		_model_component->_dE_dy_for_worker.push_back( \
				_model_component->_layers[i]->getDEDY());
		if (_model_component->_layers_param[i]->getParamTrainType() == NEED \
				&& i > 0) {
			///> 为了反向对weight和bias求导时要用到
			_model_component->_y_needed_train.push_back( \
					_model_component->_layers[i-1]->getY());
		}
	}
}


template <typename Dtype>
void TrainModel<Dtype>::createMPIDist() {
	int num_trans;
	int num_pars_type = _model_component->_num_need_train_layers;
	if(_model_component->_pid == _model_component->_master_pid)
		num_trans = _model_component->_num_process - 1;
	else
		num_trans = 1;

	MPIDistribute<Dtype> *send_recv_w[num_trans*num_pars_type];  
	MPIDistribute<Dtype> *send_recv_bias[num_trans*num_pars_type];

	if(_model_component->_pid == _model_component->_master_pid){
		for(int i = 0; i < num_trans; i++){
			for(int j = 0; j < num_pars_type; j++){
				int trans_pid = i+1;
				send_recv_w[i*num_pars_type+j] = new MPIDistribute<Dtype>( \
						_model_component->_w_len[j], i+(6+j*2)*num_trans, \
						trans_pid, MPI_FLOAT, _model_component->_w[j]->getDevData());	
				send_recv_bias[i*num_pars_type+j] = new MPIDistribute<Dtype>( \
						_model_component->_bias_len[j], i+(7+j*2)*num_trans, \
						trans_pid, MPI_FLOAT, _model_component->_bias[j]->getDevData());
				_model_component->_send_recv_w.push_back(send_recv_w[i*num_pars_type+j]);
				_model_component->_send_recv_bias.push_back(send_recv_bias[i*num_pars_type+j]);
			}
		}
	}else{
		num_trans = _model_component->_num_process - 1;
		for(int j = 0; j < num_pars_type; j++){
			int trans_pid = 0;
			send_recv_w[j] = new MPIDistribute<Dtype>( \
					_model_component->_w_len[j], this->_model_component->_pid-1+(6+j*2)*num_trans, \
					trans_pid, MPI_FLOAT, _model_component->_w[j]->getDevData());	
			send_recv_bias[j] = new MPIDistribute<Dtype>( \
					_model_component->_bias_len[j], this->_model_component->_pid-1+(7+j*2)*num_trans, \
					trans_pid, MPI_FLOAT, _model_component->_bias[j]->getDevData());
			_model_component->_send_recv_w.push_back(send_recv_w[j]);
			_model_component->_send_recv_bias.push_back(send_recv_bias[j]);
		}
	}

}

template <typename Dtype>
void TrainModel<Dtype>::initWeightAndBcastByRandom() {

	for (int k = 0; k < _model_component->_num_need_train_layers; ++k) {
		if (_model_component->_pid == _model_component->_master_pid) {
			gaussRand(_model_component->_w[k], \
					dynamic_cast<TrainParam*>( \
						_model_component->_layers_need_train_param[k])->getWGauss());
			cudaMemset(_model_component->_bias[k]->getDevData(), 0, \
					sizeof(float) * _model_component->_bias_len[k]);
		}
		MPIDistribute<Dtype> *bcast_w = new MPIDistribute<Dtype>( \
				_model_component->_w_len[k], 0, \
				0, MPI_FLOAT, _model_component->_w[k]->getDevData());
		bcast_w->bcast();
		MPIDistribute<Dtype> *bcast_bias = new MPIDistribute<Dtype>( \
				_model_component->_bias_len[k], 0, \
				0, MPI_FLOAT, _model_component->_bias[k]->getDevData());
		bcast_bias->bcast();
		delete bcast_w;
		delete bcast_bias;
	}
}

template <typename Dtype>
void TrainModel<Dtype>::initWeightAndBcastByFile(vector<string> w_file, \
		vector<string> bias_file) {
	for (int k = 0; k < _model_component->_num_need_train_layers; ++k) {
		if (_model_component->_pid == _model_component->_master_pid) {
			_model_component->_w[k]->readPars(w_file[k]);
			_model_component->_bias[k]->readPars(bias_file[k]);
		}
		MPIDistribute<Dtype> *bcast_w = new MPIDistribute<Dtype>( \
				_model_component->_w_len[k], 0, \
				0, MPI_FLOAT, _model_component->_w[k]->getDevData());
		bcast_w->bcast();
		MPIDistribute<Dtype> *bcast_bias = new MPIDistribute<Dtype>( \
				_model_component->_bias_len[k], 0, \
				0, MPI_FLOAT, _model_component->_bias[k]->getDevData());
		bcast_bias->bcast();
		delete bcast_w;
		delete bcast_bias;
	}
}

template <typename Dtype>
void TrainModel<Dtype>::forwardPropagate(){
	for (int k = 0; k < _model_component->_num_layers-1; ++k) {
		_model_component->_layers[k]->computeOutput(\
				_model_component->_y_for_worker[k]);
	}
}

template <typename Dtype>
void TrainModel<Dtype>::backwardPropagate(){
	for (int k = _model_component->_num_layers-2; k > 0; --k) {
		_model_component->_layers[k]->computeDerivsOfInput( \
				_model_component->_dE_dy_for_worker[k-1]);
	}
}

template <typename Dtype>
void TrainModel<Dtype>::computeAndUpdatePars(){
	for (int k = _model_component->_num_need_train_layers-1; k >= 0; --k) {
		TrainLayer<Dtype> *tl = dynamic_cast< TrainLayer<Dtype>* >( \
				_model_component->_layers_needed_train[k]);
		tl->computeDerivsOfPars(_model_component->_y_needed_train[k]);
		tl->updatePars();
	}
}

template <typename Dtype>
void TrainModel<Dtype>::earlyStopping(int epoch_idx) {
	if(_strip_likelihood.size() == 0){
		_min_likelihood = _likelihood;
		_min_error = _error;
		_min_epoch = epoch_idx;
		_strip_likelihood.push_back(_likelihood);
	}else if(_strip_likelihood.size() < _num_strip){
		_strip_likelihood.push_back(_likelihood);
		if(_min_likelihood > _likelihood){
			_min_likelihood = _likelihood;
			_min_error = _error;
			_min_epoch = epoch_idx;
		}
	}else if(_strip_likelihood.size() == _num_strip){
		if(_min_likelihood > _likelihood){
			_min_likelihood = _likelihood;
			_min_error = _error;
			_min_epoch = epoch_idx;
		}
		_strip_likelihood.erase(_strip_likelihood.begin());
		_strip_likelihood.push_back(_likelihood);

		double tmp = 0;
		//	std::accumulate(_strip_likelihood.begin(), _strip_likelihood.end(), tmp);

		vector<float>::iterator min_value = min_element(_strip_likelihood.begin(), _strip_likelihood.end());

		double generalization_loss = 100*(_likelihood/_min_likelihood - 1);
		double progress_loss = 1000 * (tmp / (_num_strip*(*min_value)) - 1);

		cout << generalization_loss << ":" << progress_loss << endl;

		if(generalization_loss / progress_loss > 0.8)
			_is_stop = true;
	}else{
		cerr << "early Stopping parameters are wrong." << endl;
		exit(EXIT_FAILURE);
	}
}

template <typename Dtype>
void TrainModel<Dtype>::sendAndRecvWBiasForWorker(const int epoch_idx, \
		const int batch_idx, int flag){
	if((batch_idx + 1) % this->_model_component->_n_push == 0 || this->_is_stop == true){ 
		if(this->_is_stop == true){
			flag = PROCESS_END;
		}else if(epoch_idx == this->_model_component->_num_epoch - 1){ 
			if((batch_idx + this->_model_component->_n_push) >= \
					this->_model_component->_num_train_batch \
					|| batch_idx == this->_model_component->_num_train_batch - 1)
				flag = PROCESS_END;
			else
				flag = batch_idx*2+1;
		}else{
			flag = batch_idx*2+1;
		}
#pragma omp parallel num_threads(this->_model_component->_num_need_train_layers)
		{   
			int tid = omp_get_thread_num();
			this->_model_component->_send_recv_w[tid]->sendFlag(flag);
			this->_model_component->_send_recv_bias[tid]->setFlag(flag);
			this->_model_component->_send_recv_w[tid]->dataTo();
			this->_model_component->_send_recv_bias[tid]->dataTo();
		}   
	}
	if((batch_idx + 1) % this->_model_component->_n_fetch == 0 || this->_is_stop == true){ 
		if(this->_is_stop == true){
			flag = PROCESS_END;
		}else if(epoch_idx == this->_model_component->_num_epoch - 1){ 
			if((batch_idx + this->_model_component->_n_fetch) >= \
					this->_model_component->_num_train_batch \
					|| batch_idx == this->_model_component->_num_train_batch - 1)
				flag = PROCESS_END;
			else
				flag = batch_idx*2;
		}else{
			flag = batch_idx*2;
		}
#pragma omp parallel num_threads(this->_model_component->_num_need_train_layers)
		{   
			int tid = omp_get_thread_num();
			this->_model_component->_send_recv_w[tid]->sendFlag(flag);
			this->_model_component->_send_recv_bias[tid]->setFlag(flag);
			this->_model_component->_send_recv_w[tid]->dataFrom();
			this->_model_component->_send_recv_bias[tid]->dataFrom();
		}   
	}

}






