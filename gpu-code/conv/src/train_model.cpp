///
/// \file train_model.cpp
/// @brief


#include <iostream>
#include "train_model.hpp"
#include "json/json.h"

using namespace std;

template <typename Dtype>
TrainModel<Dtype>::TrainModel(){
	_model_component = new ModelComponent<Dtype>();
	_likelihood = 0;
}

template <typename Dtype>
TrainModel<Dtype>::~TrainModel() {
	delete _model_component;
	delete _voc;
}

template <typename Dtype>
void TrainModel<Dtype>::parseImgBinary(int num_process){
	_model_component->_num_process = num_process;

	//一次要读取多个进程需要处理的数据
	_voc = new LoadVOC<Dtype>(_model_component->_minibatch_size * num_process);
	_model_component->_num_train = _voc->getNumTrain();
	_model_component->_num_valid = _voc->getNumValid();
	_model_component->_num_train_each_process = _voc->getNumTrain()/(num_process-1);
	_model_component->_num_valid_each_process = _voc->getNumValid()/(num_process-1);
	_model_component->setNumTrainBatch();
	_model_component->setNumValidBatch();

	_model_component->_one_img_len = _model_component->_img_size\
										*_model_component->_img_size \
										*_model_component->_img_channel;
	_model_component->_in_len_each_process = _model_component->_one_img_len \
 										*_model_component->_minibatch_size;
}

template <typename Dtype>
void TrainModel<Dtype>::parseNetJson(string json_file) {
	Json::Reader reader;
	Json::Value root;
	ifstream fin(json_file.c_str());
	if (reader.parse(fin, root)) {
		_model_component->_minibatch_size = root["minibatch_size"].asInt();
		Param::_minibatch_size = _model_component->_minibatch_size;
		_model_component->_n_push = root["n_push"].asInt();
		_model_component->_n_fetch = root["n_fetch"].asInt();
		_model_component->_num_epoch = root["num_epoch"].asInt();
		_model_component->_img_size = root["img_size"].asInt();
		_model_component->_img_channel = root["img_channel"].asInt();

		cout << "\n===========overall==============" \
			<< "\nnum_epoch: " << _model_component->_num_epoch \
			<< "\nbatchSize: " << _model_component->_minibatch_size \
			<< "\nn_fetch: " << _model_component->_n_fetch \
			<< "\nn_push: " << _model_component->_n_push;

		_model_component->_num_layers = root["layer"].size();

		string layer_type, name;
		int pad, stride, filter_size, filter_channel, num_out;
		float w_lr, bias_lr, momentum, weight_decay;
		string p_type;
		Param* param;

		for (int i = 0; i < _model_component->_num_layers; ++i) {
			layer_type = root["layer"][i]["type"].asString();
			name = root["layer"][i]["name"].asString();
			if (!root["layer"][i]["pad"].isNull()) {
				pad = root["layer"][i]["pad"].asInt();
				stride = root["layer"][i]["stride"].asInt();
				filter_size = root["layer"][i]["filter_size"].asInt();
			}
			if (!root["layer"][i]["w_lr"].isNull()) {
				w_lr = root["layer"][i]["w_lr"].asFloat();
				bias_lr = root["layer"][i]["bias_lr"].asFloat();
				momentum = root["layer"][i]["momentum"].asFloat();
				weight_decay = root["layer"][i]["weight_decay"].asFloat();
			}
			if (!root["layer"][i]["num_out"].isNull()) {
				num_out = root["layer"][i]["num_out"].asInt();
			}
			if (!root["layer"][i]["pool_type"].isNull()) {
				p_type = root["layer"][i]["pool_type"].asString();
			}
			if (!root["layer"][i]["filter_channel"].isNull()) {
				filter_channel = root["layer"][i]["filter_channel"].asInt();
			}else{
				filter_channel = 0;
			}
			if (!root["layer"][i]["w_gauss"].isNull()) {
				_model_component->_w_init_gauss.push_back( \
						root["layer"][i]["w_gauss"].asFloat());
			}
			if (layer_type == "CONVOLUTION") {
				if (_model_component->_layers_param.size() == 0) {
					param = new ConvParam( \
							_model_component->_string_map_layertype[layer_type], \
							name, w_lr, bias_lr, momentum, weight_decay, \
							_model_component->_img_size, pad, stride, \
							_model_component->_img_channel, filter_size, \
							filter_channel);
				} else{
					param = new ConvParam( \
							_model_component->_string_map_layertype[layer_type], \
							name, w_lr, bias_lr, momentum, weight_decay, \
							pad, stride, filter_size, filter_channel, \
							dynamic_cast<LocalConnectParam*>(_model_component->_layers_param.back()));
				}
				param->printParam();
			} else if (layer_type == "POOLING") {
				param = new PoolParam( \
							_model_component->_string_map_layertype[layer_type], \
							name, pad, stride, filter_size, 0, \
							dynamic_cast<LocalConnectParam*>(_model_component->_layers_param.back()), _model_component->_string_map_pooltype[p_type]);
				param->printParam();
			} else if (layer_type == "SIGMOID" || layer_type == "RECTIFIED" \
					|| layer_type == "SOFTMAX" || layer_type == "DROPOUT") {
				param = new FullConnectParam( \
							_model_component->_string_map_layertype[layer_type], \
							name, 0, _model_component->_layers_param.back());
				param->printParam();
			} else if (layer_type == "INNERPRODUCT" ) {
				param = new InnerParam( \
							_model_component->_string_map_layertype[layer_type], \
                            name, w_lr, bias_lr, momentum, weight_decay, \
                            num_out, _model_component->_layers_param.back());
				param->printParam();
			}
			_model_component->_layers_param.push_back(param);
		}
	}
}

template <typename Dtype>
void TrainModel<Dtype>::createLayerForWorker(){
	for (int i = 0; i < _model_component->_num_layers; ++i){
		Layer<Dtype> *layer;
		if (_model_component->_layers_param[i]->getLayerType() == CONVOLUTION) {
			layer = new ConvNet<Dtype>( \
						(ConvParam*)_model_component->_layers_param[i]);
		} else if (_model_component->_layers_param[i]->getLayerType() == POOLING) {
			layer = new PoolingLayer<Dtype>( \
						(PoolParam*)_model_component->_layers_param[i]);
		} else if (_model_component->_layers_param[i]->getLayerType() == SIGMOID) {
			layer = new SigmoidLayer<Dtype>( \
						(FullConnectParam*)_model_component->_layers_param[i]);
		} else if (_model_component->_layers_param[i]->getLayerType() == RECTIFIED) {
			layer = new ReluLayer<Dtype>( \
						(FullConnectParam*)_model_component->_layers_param[i]);
		} else if (_model_component->_layers_param[i]->getLayerType() == SOFTMAX) {
			layer = new Logistic<Dtype>( \
						(FullConnectParam*)_model_component->_layers_param[i]);
		} else if (_model_component->_layers_param[i]->getLayerType() == DROPOUT) {
			layer = new DropoutLayer<Dtype>( \
						(FullConnectParam*)_model_component->_layers_param[i]);
		} else if (_model_component->_layers_param[i]->getLayerType() \
					== INNERPRODUCT ) {
			layer = new InnerProductLayer<Dtype>( \
						(FullConnectParam*)_model_component->_layers_param[i]);
		}
		layer->initCuda();
		_model_component->_layers.push_back(layer);

		if (_model_component->_layers_param[i]->getParamTrainType() == NEED) {
			_model_component->_layers_need_train_param.push_back( \
							_model_component->_layers_param[i]);
			_model_component->_layers_needed_train.push_back(layer);
			_model_component->_num_need_train_layers++;
		}
	}
}

template <typename Dtype>
void TrainModel<Dtype>::createWBiasForManager() {
	for (int i = 0; i < _model_component->getNumNeedTrainLayers(); ++i) {
		Matrix<Dtype> *w = new Matrix<Dtype>( \
					_model_component->_layers_needed_train[i]->getW());
		Matrix<Dtype> *bias = new Matrix<Dtype>( \
					_model_component->_layers_needed_train[i]->getBias());

		_model_component->_w.push_back(w);
		_model_component->_bias.push_back(bias);
		_model_component->_w_len.push_back( \
				_model_component->_layers_needed_train[i]->getW()->getNumEles());
		_model_component->_bias_len.push_back( \
				_model_component->_layers_needed_train[i]->getBias()->getNumEles());
	}
}

template <typename Dtype>
void TrainModel<Dtype>::createWBiasForWorker() {
	for (int i = 0; i < _model_component->getNumNeedTrainLayers(); ++i) {
		_model_component->_w.push_back( \
		 			_model_component->_layers_needed_train[i]->getW());
		_model_component->_bias.push_back( \
					_model_component->_layers_needed_train[i]->getBias());
		_model_component->_w_len.push_back( \
					_model_component->_layers_needed_train[i]->getW()->getNumEles());
		_model_component->_bias_len.push_back( \
				_model_component->_layers_needed_train[i]->getBias()->getNumEles());
	}
}

template <typename Dtype>
void TrainModel<Dtype>::createYDEDYForWorker() {
	_model_component->_y_for_worker.push_back(_model_component->_mini_data);
	for (int i = 0; i < _model_component->_num_layers; ++i){
		_model_component->_y_for_worker.push_back( \
					_model_component->_layers_needed_train[i]->getY());
		_model_component->_dE_dy_for_worker.push_back( \
					_model_component->_layers_needed_train[i]->getDEDY());
		if (_model_component->_layers_param[i]->getParamTrainType() == NEED \
						&& i > 0) {
			///> 为了反向对weight和bias求导时要用到
			_model_component->_y_for_compute_par.push_back( \
					_model_component->_layers_needed_train[i-1]->getY());
		}
	}
}


template <typename Dtype>
void TrainModel<Dtype>::createVOCPixelAndLabel(){
	int times = 0;
	if (_model_component->_rank == 0) {
		times = _model_component->_num_process - 1;
	}else{
		times = 1;
	}
	_model_component->_mini_train_data = new Matrix<Dtype>( \
			_model_component->_minibatch_size*times, \
			_model_component->_one_img_len);
	_model_component->_mini_valid_data = new Matrix<Dtype>( \
			_model_component->_minibatch_size*times, \
			_model_component->_one_img_len);
	_model_component->_mini_train_label_for_voc = new Matrix< vector<int> >( \
			_model_component->_minibatch_size*times, 1);
	_model_component->_mini_valid_label_for_voc = new Matrix< vector<int> >( \
			_model_component->_minibatch_size*times, 1);
}

template <typename Dtype>
void TrainModel<Dtype>::initWeightAndBcast() {
	for (int k = 0; k < _model_component->_num_need_train_layers; ++k) {
		if (_model_component->_pid == 0) {
			gaussRand(_model_component->_w[k], _model_component->_w_init_gauss[k]);
			cudaMemset(_model_component->_bias[k]->getDevData(), 0, \
                   sizeof(float) * _model_component->_bias_len[k]);
		}
		MPI_Bcast(_model_component->_w[k]->getDevData(), \
					_model_component->_w_len[k], MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Bcast(_model_component->_bias[k]->getDevData(),  \
					_model_component->_bias_len[k], MPI_FLOAT, 0, MPI_COMM_WORLD);
	}
}

template <typename Dtype>
float TrainModel<Dtype>::forwardPropagate(){
	_model_component->layers[0].computeOutputs(_model_component->mini_data);
	for (int k = 0; k < _model_component->_num_layers; ++k) {
		_model_component->layers[k].computeOutputs( \
				_model_component->_y_for_worker[k]);
	}
	_likelihood += _model_component->layers[_model_component->_num_layers-1].computeError( \
				_model_component->_mini_label_for_voc, _error);
	return _likelihood;
}

template <typename Dtype>
void TrainModel<Dtype>::backwardPropagate(){
	_model_component->layers[_model_component->_num_layers-1].computeDerivsOfInput( \
					_model_component->_y_for_compute_par.back(), \
	              	_model_component->_mini_label_for_voc);
	for (int k = _model_component->_num_layers-2; k >= 1; --k) {
		_model_component->layers[k].computeDerivsOfInput( \
				_model_component->_dE_dy_for_worker[k-1]);
	}
}

template <typename Dtype>
void TrainModel<Dtype>::updatePars(){
	for (int k = _model_component->_num_layers-1; k >= 1; --k) {
		_model_component->train_layers[k].computeDerivsOfPars( \
				_model_component->_y_for_compute_par[k-1]);
	}
}

template <typename Dtype>
void TrainModel<Dtype>::train() {
	for (int epoch_idx = 0; epoch_idx < _model_component->_num_epoch; epoch_idx++) {
		for(int batch_idx = 0; batch_idx < _model_component->_num_train_batch; batch_idx++){
			//
		}
	}
}


















