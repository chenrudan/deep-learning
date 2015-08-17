///
/// \file train_model.cpp
/// @brief


#include <iostream>
#include "train_model.hpp"
#include "json/json.h"

using namespace std;

template <typename Dtype>
TrainModel<Dtype>::TrainModel(const int pid){
	_model_component = new ModelComponent<Dtype>();
	_model_component->_pid = pid;
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
	if(_model_component->_pid == 0){
//		_voc = new LoadVOC<Dtype>(_model_component->_minibatch_size * num_process);
		_voc = new LoadCifar10<Dtype>(_model_component->_minibatch_size);
		_model_component->_num_train = _voc->getNumTrain();
		_model_component->_num_valid = _voc->getNumValid();
		_model_component->_num_train_each_process = _voc->getNumTrain()/(num_process-1);
		_model_component->_num_valid_each_process = _voc->getNumValid()/(num_process-1);
		_model_component->setNumTrainBatch();
		_model_component->setNumValidBatch();
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
				_model_component->_pid, 0, MPI_INT, \
				&_model_component->_num_train_batch);
		MPIDistribute<int> *recv_valid = new MPIDistribute<int>(1, \
				_model_component->_pid+num_process, \
				0, MPI_INT, &_model_component->_num_valid_batch);
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
		_model_component->_img_size = root["img_size"].asInt();
		_model_component->_img_channel = root["img_channel"].asInt();

		if(_model_component->_pid == 0){
		cout << "\n===========overall==============" \
			<< "\nnum_epoch: " << _model_component->_num_epoch \
			<< "\nbatchSize: " << _model_component->_minibatch_size \
			<< "\nn_fetch: " << _model_component->_n_fetch \
			<< "\nn_push: " << _model_component->_n_push;
		}

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
							dynamic_cast<LocalConnectParam*>( \
								_model_component->_layers_param.back()));
				}
			} else if (layer_type == "POOLING") {
				param = new PoolParam( \
							_model_component->_string_map_layertype[layer_type], \
							name, pad, stride, filter_size, 0, \
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
				param = new InnerParam( \
							_model_component->_string_map_layertype[layer_type], \
                            name, w_lr, bias_lr, momentum, weight_decay, \
                            num_out, _model_component->_layers_param.back());
			}
			if(_model_component->_pid == 0)
				param->printParam();
			_model_component->_layers_param.push_back(param);

			if (param->getParamTrainType() == NEED) {
				_model_component->_layers_need_train_param.push_back(param);
				_model_component->_num_need_train_layers++;
			}
		}
	}
	_model_component->_one_img_len = _model_component->_img_size\
										*_model_component->_img_size \
										*_model_component->_img_channel;
	_model_component->_in_len_each_process = _model_component->_one_img_len \
 										*_model_component->_minibatch_size;
}

template <typename Dtype>
void TrainModel<Dtype>::createLayerForWorker(){
	for (int i = 0; i < _model_component->_num_layers; ++i){
		Layer<Dtype> *layer;
		if (_model_component->_layers_param[i]->getLayerType() == CONVOLUTION) {
			layer = new ConvNet<Dtype>( \
						dynamic_cast<ConvParam*>(_model_component->_layers_param[i]));
		} else if (_model_component->_layers_param[i]->getLayerType() == POOLING) {
			layer = new PoolingLayer<Dtype>( \
						dynamic_cast<PoolParam*>(_model_component->_layers_param[i]));
		} else if (_model_component->_layers_param[i]->getLayerType() == SIGMOID) {
			layer = new SigmoidLayer<Dtype>( \
						dynamic_cast<FullConnectParam*>(_model_component->_layers_param[i]));
		} else if (_model_component->_layers_param[i]->getLayerType() == RECTIFIED) {
			layer = new ReluLayer<Dtype>( \
						dynamic_cast<FullConnectParam*>(_model_component->_layers_param[i]));
		} else if (_model_component->_layers_param[i]->getLayerType() == SOFTMAX) {
			layer = new Logistic<Dtype>( \
						dynamic_cast<FullConnectParam*>(_model_component->_layers_param[i]));
		} else if (_model_component->_layers_param[i]->getLayerType() == DROPOUT) {
			layer = new DropoutLayer<Dtype>( \
						dynamic_cast<FullConnectParam*>(_model_component->_layers_param[i]));
		} else if (_model_component->_layers_param[i]->getLayerType() \
					== INNERPRODUCT ) {
			layer = new InnerProductLayer<Dtype>( \
						dynamic_cast<InnerParam*>(_model_component->_layers_param[i]));
		}
		layer->initCuda();
		_model_component->_layers.push_back(layer);

		if (_model_component->_layers_param[i]->getParamTrainType() == NEED) {
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
			w = new Matrix<Dtype>(cp->getFilterSize() \
				*cp->getFilterSize()*cp->getInChannel(), cp->getOutChannel());
			w_len = cp->getFilterSize()*cp->getFilterSize()\
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
	for (int i = 0; i < _model_component->_num_layers; ++i){
		_model_component->_y_for_worker.push_back( \
					_model_component->_layers[i]->getY());
		_model_component->_dE_dy_for_worker.push_back( \
					_model_component->_layers[i]->getDEDY());
		if (_model_component->_layers_param[i]->getParamTrainType() == NEED \
						&& i > 0) {
			///> 为了反向对weight和bias求导时要用到
			_model_component->_y_for_compute_par.push_back( \
					_model_component->_layers[i-1]->getY());
		}
	}
}


template <typename Dtype>
void TrainModel<Dtype>::createPixelAndLabel(){
	int times = 0;
	if (_model_component->_pid == 0) 
		times = _model_component->_num_process - 1;
	else
		times = 1;
	
	for(int i = 0; i < times*2; i++){
		Matrix<Dtype> *pixel = new Matrix<Dtype>( \
					_model_component->_minibatch_size*times, \
					_model_component->_one_img_len);
		Matrix<int> *label = new Matrix<int>( \
					_model_component->_minibatch_size*times, 1);
		_model_component->_mini_data.push_back(pixel);
		_model_component->_mini_label.push_back(label);
	}
}

template <typename Dtype>
void TrainModel<Dtype>::createLabelNum(){
	int times = 0;
	if (_model_component->_pid == 0) {
		times = _model_component->_num_process - 1;
	}else{
		times = 1;
	}
	_model_component->_mini_train_label_num = new Matrix<int>( \
			_model_component->_minibatch_size*times, 1);
	_model_component->_mini_valid_label_num = new Matrix<int>( \
			_model_component->_minibatch_size*times, 1);
}

template <typename Dtype>
void TrainModel<Dtype>::initWeightAndBcast() {
	cout << _model_component->_pid << endl;
	cout << _model_component->_num_need_train_layers << endl;

	for (int k = 0; k < _model_component->_num_need_train_layers; ++k) {
		if (_model_component->_pid == 0) {
			gaussRand(_model_component->_w[k], _model_component->_w_init_gauss[k]);
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

	MPIDistribute<Dtype> *recv_train_pixel = new MPIDistribute<Dtype>( \
			_model_component->_minibatch_size*_model_component->_one_img_len, \
			_model_component->_pid-1, 0, MPI_FLOAT, \
			_model_component->_mini_data[0]->getDevData());
	MPIDistribute<int> *recv_train_label = new MPIDistribute<int>(\
			_model_component->_minibatch_size, \
			_model_component->_pid-1 + (_model_component->_num_process-1), \
			0, MPI_INT, _model_component->_mini_label[0]->getDevData());
	MPIDistribute<Dtype> *recv_valid_pixel = new MPIDistribute<Dtype>( \
			_model_component->_minibatch_size*_model_component->_one_img_len, \
			_model_component->_pid-1 + (_model_component->_num_process-1)*2, \
			0, MPI_FLOAT, _model_component->_mini_data[1]->getDevData());
	MPIDistribute<int> *recv_valid_label = new MPIDistribute<int>(\
			_model_component->_minibatch_size, \
			_model_component->_pid-1 + (_model_component->_num_process-1)*3, \
			0, MPI_INT, _model_component->_mini_label[1]->getDevData());

	for (int epoch_idx = 0; epoch_idx < _model_component->_num_epoch; \
					epoch_idx++) {
		for(int batch_idx = 0; batch_idx < _model_component->_num_train_batch; \
						batch_idx++){
			recv_train_pixel->sendFlag(batch_idx);
			recv_train_pixel->dataFrom();
			recv_train_label->dataFrom();
/*
		//一个线程传递数据，一个线程执行运算
		#pragma omp parallel num_threads(2)
			{
				int tid = omp_get_thread_num();
				if(tid == 0){
					recv_pixel->sendFlag(batch_idx);
					recv_pixel->dataFrom();
					recv_label->sendFlag(batch_idx);
					recv_label->dataFrom();
				}else{
				//
				}
			}
	*/
		}	
	}
	delete recv_train_pixel;
	delete recv_valid_pixel;
	delete recv_train_label;
	delete recv_valid_label;
}


template <typename Dtype>
void TrainModel<Dtype>::sendPixelAndLabel() {
	
	int num_send = _model_component->_num_process - 1;

	MPIDistribute<Dtype> *send_pixel[2*num_send];  //2表示train和valid,相邻的是
												//train和valid，然后是下一个进程的
	MPIDistribute<int> *send_label[2*num_send];

	int pixel_len = _model_component->_minibatch_size*_model_component->_one_img_len;
	int label_len = _model_component->_minibatch_size;

	for(int i = 0; i < num_send; i++){

		send_pixel[i*2] = new MPIDistribute<Dtype>( \
				pixel_len, i, i+1, MPI_FLOAT, \
				_model_component->_mini_data[i*2]->getDevData());	
		send_label[i*2] = new MPIDistribute<int>( \
				label_len, i+num_send, i+1, MPI_INT, \
				_model_component->_mini_label[i*2]->getDevData());
		send_pixel[i*2+1] = new MPIDistribute<Dtype>( \
				pixel_len, i+num_send*2, i+1, MPI_FLOAT, \
				_model_component->_mini_data[i*2+1]->getDevData());
		send_label[i*2+1] = new MPIDistribute<int>( \
				label_len, i+num_send*3, i+1, MPI_INT, \
				_model_component->_mini_label[i*2+1]->getDevData());
	}
	Dtype *h_mini_pixel;   //分配在主机内存上
	int *h_mini_label;     
	#pragma omp parallel num_threads(num_send*2)
	{
		int tid = omp_get_thread_num();
		int pid = tid / 2 + 1;   //计算出对应的进程ID
		int type_id = tid % 2;   //计算是train还是valid
		while(send_pixel[tid]->getFlag() != PROCESS_END){
			send_pixel[tid]->receviceFlag();
			if(type_id == 0)
				_voc->loadTrainOneBatch(0, num_send, pid-1, \
							h_mini_pixel, h_mini_label);
			else
				_voc->loadValidOneBatch(0, num_send, pid-1, \
							h_mini_pixel, h_mini_label);
			_model_component->_mini_data[tid]->copyFromHost(h_mini_pixel, pixel_len);
			_model_component->_mini_label[tid]->copyFromHost(h_mini_label, label_len);

			send_pixel[tid]->dataTo();
			send_label[tid]->dataTo();
		}
	}		
	delete[] send_pixel;
	delete[] send_label;
}













