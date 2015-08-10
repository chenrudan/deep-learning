///
/// \file param.h
///
#ifndef PARAM_H_
#define PARAM_H_

#include <string>
#include <iostream>
#include <cmath>

using namespace std;

#define MAX_THREAD_SIZE 32
#define MAX_NUM_KERNEL 4096
#define MAX_NUM_THREAD 1024

typedef enum PARAM_CONNECT_TYPE {
    PARAM_CONNECT_TYPE_LOCAL = 0,
    PARAM_CONNECT_TYPE_FULL = 1
} ConnectType;

typedef enum POOLING_TYPE {
	MAX_POOLING = 0,
	AVG_POOLING = 1
} PoolingType;

/// \brief 实现了每一层的参数
///
class Param {

public:
    Param() { }

    virtual ~Param() { }

    Param(string name, const int minibatch_size) : _name(name), \
							_minibatch_size(minibatch_size){}

	inline virtual int getNumOut() {return 0;}
	inline virtual int getOutChannel() {return 0;}
	inline virtual int getOutSize() {return 0;}
	

    inline int getMinibatchSize() {
        return _minibatch_size;
    }
	inline string getName(){
		return _name;
	}
    inline ConnectType getConnectType() {
        return type;
    }

protected:
    string _name;  ///> 实例化每一层的名字，用来区分不同的层
    int _minibatch_size;
    ConnectType type;
};

/// \brief 实现了需要训练的层参数，主要为了改变权重和调节学习率
class TrainParam : public virtual Param {
public:
    TrainParam() { }

    virtual ~TrainParam() { }

    TrainParam(const float w_lr, const float b_lr, \
            const float momentum, const float weight_decay, \
			const int n_push, const int n_fetch) \
		: _w_lr(w_lr), _b_lr(b_lr), _momentum(momentum), \
		_weight_decay(w_lr*weight_decay), _n_push(n_push), \
		_n_fetch(n_fetch){}

    inline void lrMultiScale(float lr_scale) {
        _w_lr *= lr_scale;
        _b_lr *= lr_scale;
		cout << _w_lr << ":" << _b_lr << endl;
    }
    inline void lrChangeTo(float new_w, float new_b) {
        _w_lr = new_w;
        _b_lr = new_b;
    }
    inline float getWLR() {
        return _w_lr;
    }
    inline float getBiasLR() {
        return _b_lr;
    }
    inline float getMomentum() {
        return _momentum;
    }
	inline float getWeightDecay() {
		return _weight_decay;
	}
    inline int getNPush() {
        return _n_push;
    }
    inline int getNFetch() {
        return _n_fetch;
    }

protected:
    float _w_lr;
    float _b_lr;
    float _momentum;
	float _weight_decay;

    int _n_push;
    int _n_fetch;
};

/// \brief 局部连接层的参数，以图片形式保存数据
class LocalConnectParam : public virtual Param {
public:

    LocalConnectParam() { }

    virtual ~LocalConnectParam() { }

    LocalConnectParam(string name, \
		const int minibatch_size, const int in_size, \
		const int pad, const int stride, const int in_channel, \
		const int filter_size, const int out_channel) \
		: _in_size(in_size), _stride(stride), _in_channel(in_channel), \
		_pad(pad), _filter_size(filter_size), _out_channel(out_channel){

			this->_name = name;
			this->_minibatch_size = minibatch_size;
			this->type = PARAM_CONNECT_TYPE_LOCAL;
			_padded_in_size = in_size + 2 * pad;
			_out_size = ceil(((_padded_in_size - filter_size)*1.0f) / stride) + 1;
		}

    LocalConnectParam(string name, \
		const int pad, const int stride, \
		const int filter_size, const int filter_channel, \
		LocalConnectParam* lc_par) \
		: _in_size(lc_par->getOutSize()), _stride(stride), \
		_in_channel(lc_par->getOutChannel()), _pad(pad), \
		_filter_size(filter_size) {
			
			this->_name = name;
			this->_minibatch_size = lc_par->getMinibatchSize();
		
			if(filter_channel != 0)
				_out_channel = filter_channel;
			else
				_out_channel = _in_channel;

			this->type = PARAM_CONNECT_TYPE_LOCAL;
			_padded_in_size = _in_size + 2 * pad;
			_out_size = ceil(((_padded_in_size - filter_size)*1.0f) / stride) + 1;

		}



    inline int getInSize() {
        return _in_size;
    }
    inline int getInChannel() {
        return _in_channel;
    }
    inline int getOutSize() {
        return _out_size;
    }
    inline int getFilterSize() {
        return _filter_size;
    }
    inline int getOutChannel() {
        return _out_channel;
    }
    inline int getPaddedInSize() {
        return _padded_in_size;
    }

    inline int getStride(){
        return _stride;
    }

    inline int getPad(){
        return _pad;
    }

private:
    int _in_size;
    int _pad;
    int _padded_in_size;
    int _stride;
    int _in_channel;
    int _filter_size; ///>在卷积中是filter，在pooling中是pool
    int _out_size;
    int _out_channel;
};

/// \brief 全连接层的参数，展开图片为一个矢量保存数据
///
/// 可以针对每一个值做某种操作，例如Relu、sigmoid、tanh等，
/// 此处不需要训练
class FullConnectParam : public virtual Param {
public:
    FullConnectParam() { }
    virtual ~FullConnectParam() { }
    FullConnectParam(string name, \
		const int minibatch_size, const int num_in, \
		const int num_out) \
		: _num_in(num_in), _num_out(num_out) {
			this->_name = name;
			this->_minibatch_size = minibatch_size;
			this->type = PARAM_CONNECT_TYPE_FULL;
		}
    FullConnectParam(string name, \
		const int num_out, Param* par){
			this->_name = name;
			this->_minibatch_size = par->getMinibatchSize();
			this->type = PARAM_CONNECT_TYPE_FULL;
			
			///由传递进来的层类型决定计算方式
			ConnectType ct = par->getConnectType();
			if(ct == PARAM_CONNECT_TYPE_LOCAL)
				_num_in = pow(par->getOutSize(), 2) * par->getOutChannel(); 
			else if(ct == PARAM_CONNECT_TYPE_FULL)
				_num_in = par->getNumOut(); 
	
			if(num_out != 0)
				_num_out = num_out;
			else
				_num_out = _num_in;
		}


    inline int getNumIn() {
        return _num_in;
    }

    inline int getNumOut() {
        return _num_out;
    }

private:
    int _num_in;
    int _num_out;
};

class ConvParam : public TrainParam, public LocalConnectParam {
public:
    ConvParam(){}

    ~ConvParam(){}

    ConvParam(const string name, const int minibatch_size, \
            const float w_lr, const float b_lr, const float momentum, \
			const float weight_decay, \
            const int n_push, const int n_fetch, const int in_size, \
            const int pad, const int stride, const int in_channel, \
            const int filter_size, const int filter_channel) \
            : TrainParam(w_lr, b_lr, momentum, weight_decay, n_push, n_fetch), \
              LocalConnectParam(name, minibatch_size, in_size, \
		            pad, stride, in_channel, filter_size, filter_channel) {}

    ConvParam(const string name, const float w_lr, \
            const float b_lr, const float momentum, \
			const float weight_decay, \
            const int n_push, const int n_fetch, const int pad, \
            const int stride, const int filter_size, \
            const int filter_channel, LocalConnectParam *lc_par) \
            : TrainParam(w_lr, b_lr, momentum, weight_decay, \
				   	n_push, n_fetch), \
              LocalConnectParam(name, pad, stride, \
		            filter_size, filter_channel, lc_par)  {}
};

class PoolParam : public LocalConnectParam {
public:
		PoolParam() {}
		~PoolParam() {}

    	PoolParam(const string name, const int minibatch_size, \
            const int in_size, const int pad, const int stride, \
			const int in_channel, const int filter_size, \
			const int filter_channel, PoolingType p_type) 
            :  LocalConnectParam(name, minibatch_size, in_size, \
					pad, stride, in_channel, filter_size, filter_channel) , \
			_p_type(p_type){
				_box_num_size = ceil((this->getOutSize() - MAX_THREAD_SIZE) \
							* 1.0f / MAX_THREAD_SIZE) + 1;
				_box_in_size = (MAX_THREAD_SIZE - 1) * stride + filter_size;
			}
			


    	PoolParam(const string name, const int pad, const int stride, \
			const int filter_size, const int filter_channel, \
			LocalConnectParam* lc_par, PoolingType p_type) 
            :  LocalConnectParam(name, pad, stride, filter_size, \
					filter_channel, lc_par), \
			_p_type(p_type){
				_box_num_size = ceil((this->getOutSize() - MAX_THREAD_SIZE) \
							* 1.0f / MAX_THREAD_SIZE) + 1;
				_box_in_size = (MAX_THREAD_SIZE - 1) * stride + filter_size;
			}


		inline PoolingType getPoolType(){
			return _p_type;
		}
		inline int getBoxNumSize(){
			return _box_num_size;
		}
		inline int getBoxInSize(){
			return _box_in_size;
		}
		


private:
	PoolingType _p_type;	
	int _box_in_size; ///>用来计算一个box输出的卷积输入
	int _box_num_size;  ///>总的box个数的行/列 
};


/// \brief 可以进行训练的全连接层
class InnerParam : public TrainParam, public FullConnectParam {
public:
    InnerParam(){}

    ~InnerParam() {}

    InnerParam(const string name, const int minibatch_size, \
		const float w_lr, const float b_lr, const float momentum, \
		const float weight_decay, \
		const int n_push, const int n_fetch, const int num_in, \
		const int num_out) \
        : TrainParam(w_lr, b_lr, momentum, weight_decay, \
				n_push, n_fetch),
          FullConnectParam(name, minibatch_size, num_in, num_out){}

    InnerParam(const string name, const float w_lr, const float b_lr, \
        const float momentum, const float weight_decay, \
		const int n_push, const int n_fetch, \
        const int num_out, Param* par) \
        : TrainParam(w_lr, b_lr, momentum, weight_decay, \
				n_push, n_fetch),  \
          FullConnectParam(name, num_out, par) {}

};

#endif
