///
/// \file train_model.cpp
/// @brief


#include <iostream>
#include "train_model.hpp"
#include "json/json.h"
#include "inner_product_layer.hpp"
#include "logistic.hpp"
#include "sigmoid_layer.hpp"
#include "relu_layer.hpp"
#include "convnet.hpp"
#include "pooling_layer.hpp"
#include "dropout_layer.hpp"

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

		if(_model_component->_pid == 1){
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
			if(_model_component->_pid == 1)
				param->printParam();
			_model_component->_layers_param.push_back(param);

			if (param->getParamTrainType() == NEED) {
				_model_component->_layers_need_train_param.push_back(param);
				_model_component->_num_need_train_layers++;
			}
		}
	}
	_model_component->_one_img_len = _model_component->_img_size \
									 *_model_component->_img_size \
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
			}
		}catch(int e){
			cout << "dynamic point is null\n";
		}

		cout << i << endl;
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
void TrainModel<Dtype>::createPixelAndLabel(){
	int times = 0;
	if (_model_component->_pid == 0) 
		times = (_model_component->_num_process - 1);
	else
		times = 1;
	for(int i = 0; i < times*2; i++){
		Matrix<Dtype> *pixel = new Matrix<Dtype>(_model_component->_minibatch_size, \
				_model_component->_one_img_len);
		Matrix<int> *label = new Matrix<int>(_model_component->_minibatch_size, 1);
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
void TrainModel<Dtype>::createMPIDist() {
	int num_trans;
	int num_pars_type = _model_component->_num_need_train_layers;
	if(_model_component->_pid == 0)
		num_trans = _model_component->_num_process - 1;
	else
		num_trans = 1;

	MPIDistribute<Dtype> *send_pixel[2*num_trans];  //2表示train和valid,相邻的是
	//train和valid，然后是下一个进程的
	MPIDistribute<int> *send_label[2*num_trans];
	int pixel_len = _model_component->_minibatch_size*_model_component->_one_img_len;
	int label_len = _model_component->_minibatch_size;

	for(int i = 0; i < num_trans; i++){
		int trans_pid = 0;
		if(_model_component->_pid == 0)
			trans_pid = i+1;

		send_pixel[i*2] = new MPIDistribute<Dtype>( \
				pixel_len, i, trans_pid, MPI_FLOAT, \
				_model_component->_mini_data[i*2]->getDevData());	
		send_label[i*2] = new MPIDistribute<int>( \
				label_len, i+num_trans, trans_pid, MPI_INT, \
				_model_component->_mini_label[i*2]->getDevData());
		send_pixel[i*2+1] = new MPIDistribute<Dtype>( \
				pixel_len, i+num_trans*2, trans_pid, MPI_FLOAT, \
				_model_component->_mini_data[i*2+1]->getDevData());
		send_label[i*2+1] = new MPIDistribute<int>( \
				label_len, i+num_trans*3, trans_pid, MPI_INT, \
				_model_component->_mini_label[i*2+1]->getDevData());

		_model_component->_send_recv_pixel.push_back(send_pixel[i*2]);
		_model_component->_send_recv_pixel.push_back(send_pixel[i*2+1]);
		_model_component->_send_recv_label.push_back(send_label[i*2]);
		_model_component->_send_recv_label.push_back(send_label[i*2+1]);
	}

	MPIDistribute<Dtype> *send_recv_w[num_trans*num_pars_type];  
	MPIDistribute<Dtype> *send_recv_bias[num_trans*num_pars_type];

	for(int i = 0; i < num_trans; i++){
		for(int j = 0; j < num_pars_type; j++){
			int trans_pid = 0;
			if(_model_component->_pid == 0)
				trans_pid = i+1;
			send_recv_w[i*num_pars_type+j] = new MPIDistribute<Dtype>( \
					_model_component->_w_len[j], i+(4+j*2)*num_trans, \
					trans_pid, MPI_FLOAT, _model_component->_w[j]->getDevData());	
			send_recv_bias[i*num_pars_type+j] = new MPIDistribute<Dtype>( \
					_model_component->_bias_len[j], i+(5+j*2)*num_trans, \
					trans_pid, MPI_FLOAT, _model_component->_bias[j]->getDevData());
			_model_component->_send_recv_w.push_back(send_recv_w[i*num_pars_type+j]);
			_model_component->_send_recv_bias.push_back(send_recv_bias[i*num_pars_type+j]);
		}
	}
}


template <typename Dtype>
void TrainModel<Dtype>::initWeightAndBcast() {

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
	for (int k = 0; k < _model_component->_num_layers; ++k) {
		_model_component->_layers[k]->computeOutput(\
				_model_component->_y_for_worker[k]);
	}
	_likelihood += dynamic_cast<Logistic<Dtype>* >( \
			_model_component->_layers[_model_component->_num_layers-1]) \
				   ->computeError(_model_component->_mini_label_for_compute, _error);
	return _likelihood;
}

template <typename Dtype>
void TrainModel<Dtype>::backwardPropagate(){
	Logistic<Dtype> *last_layer = dynamic_cast<Logistic<Dtype>* >( \
			_model_component->_layers[_model_component->_num_layers-1]);
	last_layer->computeDerivsOfInput(_model_component->_dE_dy_for_worker[ \
			_model_component->_num_layers-2], \
			_model_component->_mini_label_for_compute);
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
void TrainModel<Dtype>::train() {

	int flag = 0;
	clock_t t;
	t = clock();
	for (int epoch_idx = 0; epoch_idx < _model_component->_num_epoch; \
			epoch_idx++) {
		_model_component->_y_for_worker[0] = _model_component->_mini_data[0];
		_model_component->_mini_label_for_compute= _model_component->_mini_label[0];
		for(int batch_idx = 0; batch_idx < _model_component->_num_train_batch; \
				batch_idx++){
			if(epoch_idx == _model_component->_num_epoch - 1 \
					&& batch_idx == _model_component->_num_train_batch - 1)
				flag = PROCESS_END;
			else
				flag = batch_idx;

			_model_component->_send_recv_pixel[0]->sendFlag(flag);
			_model_component->_send_recv_label[0]->setFlag(flag);
			_model_component->_send_recv_pixel[0]->dataFrom();
			_model_component->_send_recv_label[0]->dataFrom();

			cout << "forward\n";
			forwardPropagate();			
			cout << "backward\n";
			backwardPropagate();
			cout << "update\n";
			computeAndUpdatePars();

			if((batch_idx + 1) % _model_component->_n_push == 0){ 
				if(epoch_idx == _model_component->_num_epoch - 1){ 
					if((batch_idx + _model_component->_n_push) >= \
							_model_component->_num_train_batch \
							|| batch_idx == _model_component->_num_train_batch - 1)
						flag = PROCESS_END;
					else
						flag = batch_idx*2+1;
				}   
				else
					flag = batch_idx*2+1;
#pragma omp parallel num_threads(_model_component->_num_need_train_layers)
				{   
					int tid = omp_get_thread_num();
					_model_component->_send_recv_w[tid]->sendFlag(flag);
					_model_component->_send_recv_bias[tid]->setFlag(flag);
					_model_component->_send_recv_w[tid]->dataTo();
					_model_component->_send_recv_bias[tid]->dataTo();
				}   
			}
			if((batch_idx + 1) % _model_component->_n_fetch == 0){ 
				if(epoch_idx == _model_component->_num_epoch - 1){ 
					if((batch_idx + _model_component->_n_fetch) >= \
							_model_component->_num_train_batch \
							|| batch_idx == _model_component->_num_train_batch - 1)
						flag = PROCESS_END;
					else
						flag = batch_idx*2;
				}   
				else
					flag = batch_idx*2;
#pragma omp parallel num_threads(_model_component->_num_need_train_layers)
				{   
					int tid = omp_get_thread_num();
					_model_component->_send_recv_w[tid]->sendFlag(flag);
					_model_component->_send_recv_bias[tid]->setFlag(flag);
					_model_component->_send_recv_w[tid]->dataFrom();
					_model_component->_send_recv_bias[tid]->dataFrom();
				}   
			}

			if(batch_idx == _model_component->_num_train_batch-1){
				_model_component->_y_for_worker[0] = _model_component->_mini_data[1];
				_model_component->_mini_label_for_compute \
					= _model_component->_mini_label[1];
				Logistic<Dtype> *last_layer = dynamic_cast<Logistic<Dtype>* >( \
						_model_component->_layers[_model_component->_num_layers-1]);
				last_layer->setRecordToZero();
				double valid_likelihood = 0;
				for(int valid_idx = 0; \
						valid_idx < _model_component->_num_valid_batch; \
						valid_idx++){
					if(epoch_idx == _model_component->_num_epoch - 1 \
							&& valid_idx == _model_component->_num_valid_batch - 1)
						flag = PROCESS_END;
					else
						flag = valid_idx;

					_model_component->_send_recv_pixel[1]->sendFlag(flag);
					_model_component->_send_recv_label[1]->setFlag(flag);
					_model_component->_send_recv_pixel[1]->dataFrom();
					_model_component->_send_recv_label[1]->dataFrom();

					valid_likelihood += forwardPropagate();			
					backwardPropagate();

				}
				Matrix<int>* valid_record = last_layer->getResultRecord();
				valid_record->showValue("valid record");
			}
		}
		if(_model_component->_pid == 1){
			t = clock() - t;
			cout << ((float)t/CLOCKS_PER_SEC) << "s.\n";
			t = clock();
		}
	}
}

template <typename Dtype>
void TrainModel<Dtype>::sendAndRecvForManager() {
	int num_pars_type = _model_component->_num_need_train_layers;
	int	num_trans = _model_component->_num_process - 1;

	int pixel_len = _model_component->_minibatch_size*_model_component->_one_img_len;
	int label_len = _model_component->_minibatch_size;

	int num_threads = num_trans*num_pars_type+num_trans*2;
	cout << "num_threads: " << num_threads<< endl;
#pragma omp parallel num_threads(num_threads)
	{
		int tid = omp_get_thread_num();
	   Dtype *h_mini_pixel;   //分配在主机内存上
	   int *h_mini_label; 
	   h_mini_pixel = new Dtype[pixel_len];   //分配在主机内存上
	   h_mini_label = new int[label_len]; 
		if(tid < num_trans*2){
			int pid = tid / 2 + 1;   //计算出对应的进程ID
			int type_id = tid % 2;   //计算是train还是valid
			do{
				/*
				if(type_id == 0)
					_voc->loadTrainOneBatch(_model_component->_send_recv_pixel[tid]->getFlag()+1, \
							num_trans, pid-1, h_mini_pixel, h_mini_label);
				else
					_voc->loadValidOneBatch(_model_component->_send_recv_pixel[tid]->getFlag()+1, \
							num_trans, pid-1, h_mini_pixel, h_mini_label);
				*/

				_model_component->_mini_data[tid]->copyFromHost(h_mini_pixel, pixel_len);
				_model_component->_mini_label[tid]->copyFromHost(h_mini_label, label_len);
				_model_component->_send_recv_pixel[tid]->receviceFlag();
				_model_component->_send_recv_label[tid]->setFlag(_model_component->_send_recv_pixel[tid]->getFlag());
				_model_component->_send_recv_pixel[tid]->dataTo();
				_model_component->_send_recv_label[tid]->dataTo();

				if(_model_component->_send_recv_pixel[tid]->getFlag() == _model_component->_minibatch_size-1)
					_model_component->_send_recv_pixel[tid]->setFlag(-1);
			}while(_model_component->_send_recv_pixel[tid]->getFlag() != PROCESS_END);

		}else{
			tid -= 2*num_trans;

			cout << "w tid: "<<  tid << endl;
			do{
				_model_component->_send_recv_w[tid]->receviceFlag();
				_model_component->_send_recv_bias[tid]->setFlag(_model_component->_send_recv_w[tid]->getFlag());
				//偶数是子进程向0号请求，奇数是子进程发送给0号
				if(_model_component->_send_recv_w[tid]->getFlag() % 2 == 0){
					_model_component->_send_recv_w[tid]->dataTo();
					_model_component->_send_recv_bias[tid]->dataTo();
				}else{
					_model_component->_send_recv_w[tid]->dataFrom();
					_model_component->_send_recv_bias[tid]->dataFrom();
					//	cout << _model_component->_send_recv_w[tid]->getFlag() << endl;
				}
			}while(_model_component->_send_recv_w[tid]->getFlag() != PROCESS_END);
		}
	}
}	

