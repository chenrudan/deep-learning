///
/// \file param.cpp
/// 

#include "param.h"

using namespace std;

Param::Param(string name, const int minibatch_size){
	this->_name = name;
	this->_minibatch_size = minibatch_size;
}

TrainParam::TrainParam(const float w_lr, const float b_lr, \
		const float momentum, const float weight_decay, \
		const int n_push, const int n_fetch){
	this->_w_lr = w_lr;
	this->_b_lr = b_lr;
	this->_momentum = momentum;
	this->_n_push = n_push;
	this->_n_fetch = n_fetch;
}

LocalConnectParam::LocalConnectParam(string name, \
		const int minibatch_size, const int in_size, \
		const int pad, const int stride, const int in_channel, \
		const int filter_size, const int out_channel) \
		: Param(name, minibatch_size){
    type = PARAM_CONNECT_TYPE_LOCAL;
	this->_in_size = in_size;
	this->_stride = stride;
	this->_in_channel = in_channel;
	this->_pad = pad;
	this->_filter_size = filter_size;
	this->_out_channel = out_channel;

	this->_padded_in_size = this->_in_size + 2 * this->_pad;
	this->_out_size = ceil(((this->_in_size - this->_filter_size)*1.0f) \
			/ this->_stride) + 1;
}

LocalConnectParam::LocalConnectParam(string name, \
		const int pad, const int stride, \
		const int filter_size, const int filter_channel, \
		LocalConnectParam* lc_par)
		: Param(name, lc_par->getMinibatchSize()){
    type = PARAM_CONNECT_TYPE_LOCAL;
	this->_in_size = lc_par->getOutSize();
	this->_stride = stride;
	this->_in_channel = lc_par->getOutChannel();
	this->_pad = pad;
	this->_filter_size = filter_size;
	this->_out_channel = filter_channel;

	this->_padded_in_size = this->_in_size + 2 * this->_pad;
	this->_out_size = ceil(((this->_in_size - this->_filter_size)*1.0f) \
			/ this->_stride) + 1;
}

FullConnectParam::FullConnectParam(string name, \
		const int minibatch_size, const int num_in, \
		const int num_out) \
		: Param(name, minibatch_size){
    type = PARAM_CONNECT_TYPE_FULL;
	this->_num_in = num_in;
	this->_num_out = num_out;
}

FullConnectParam::FullConnectParam(string name, \
		const int num_out, Param* par) \
		: Param(name, par->getMinibatchSize()){
    type = PARAM_CONNECT_TYPE_FULL;
	this->_num_out = num_out;

	///由传递进来的层类型决定计算方式
	ConnectType ct = par->getConnectType();
	if(ct == PARAM_CONNECT_TYPE_LOCAL)
		this->_num_in = pow(par->getOutSize(), 2) * par->getOutChannel(); 
	else if(ct == PARAM_CONNECT_TYPE_FULL)
		this->_num_in = par->getNumOut(); 
}















