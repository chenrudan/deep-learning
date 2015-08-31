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

typedef enum PARAM_TRAIN_TYPE {
    NOTNEED = 0,
    NEED = 1
} ParamTrainType;

typedef enum LAYER_TYPE {
    CONVOLUTION = 0,
    POOLING = 1,
    SIGMOID = 2,
    RECTIFIED = 3,
    INNERPRODUCT = 4,
    SOFTMAX = 5,
    DROPOUT = 6,
	PREDICTOBJECT = 7
} LayerType;

/// \brief 实现了每一层的参数
///
class Param {

public:
    Param() { }

    virtual ~Param() { }

    Param(string name, LayerType layer_type) : \
                _name(name), _layer_type(layer_type), \
				_param_train_type(NOTNEED){}

	virtual int getNumOut() {return 0;}
	virtual int getOutChannel() {return 0;}
	virtual int getOutWidth() {return 0;}	
	virtual int getOutHeight() {return 0;}	

    inline int getMinibatchSize() {
        return _minibatch_size;
    }
	inline string getName(){
		return _name;
	}
    inline ConnectType getConnectType() {
        return type;
    }
    ParamTrainType getParamTrainType(){
        return _param_train_type;
    }
    LayerType getLayerType(){
        return _layer_type;
    }
    virtual void printParam(){
        cout << "\n============"<< _name << "============" \
                << "\nlayer_type: " << _layer_type;
    }
	static void setMinibatchSize(const int minibatch_size){
		_minibatch_size = minibatch_size;
	}

protected:
    string _name;  ///> 实例化每一层的名字，用来区分不同的层
    static int _minibatch_size;
    ConnectType type;
    ParamTrainType _param_train_type;
    LayerType _layer_type;
};

/// \brief 实现了需要训练的层参数，主要为了改变权重和调节学习率
class TrainParam : public virtual Param {
public:
    TrainParam() { }

    virtual ~TrainParam() { }

    TrainParam(const float w_lr, const float b_lr, \
            const float momentum, const float weight_decay) \
		: _w_lr(w_lr), _b_lr(b_lr), _momentum(momentum), \
		_weight_decay(w_lr*weight_decay){	
		this->_param_train_type = NEED;
	}

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
    void printParam(){
        cout << "\nw_lr: " << _w_lr \
				<< "\nb_lr: " << _b_lr \
				<< "\nmomentum: " << _momentum \
				<< "\nweight_decay: " << _weight_decay;
    }

protected:
    float _w_lr;
    float _b_lr;
    float _momentum;
	float _weight_decay;

};

/// \brief 局部连接层的参数，以图片形式保存数据
class LocalConnectParam : public virtual Param {
public:

    LocalConnectParam() { }

    virtual ~LocalConnectParam() { }

    LocalConnectParam(LayerType layer_type, string name, const int in_height, \
		const int in_width, const int pad_height, const int pad_width, \
		const int stride_height, const int stride_width, \
		const int in_channel, \
		const int filter_height, const int filter_width, const int out_channel) \
		: _in_height(in_height), _in_width(in_width), _stride_height(stride_height), \
		_stride_width(stride_width), _in_channel(in_channel), \
		_pad_height(pad_height), _pad_width(pad_width), \
		_filter_height(filter_height), _filter_width(filter_width), \
		_out_channel(out_channel){

            this->_layer_type = layer_type;
			this->_name = name;
			this->type = PARAM_CONNECT_TYPE_LOCAL;
			_padded_in_height = in_height + 2 * pad_height;
			_padded_in_width = in_width + 2 * pad_width;
			_out_height = ceil(((_padded_in_height - filter_height)*1.0f) / stride_height) + 1;
			_out_width = ceil(((_padded_in_width - filter_width)*1.0f) / stride_width) + 1;
			_box_num_height = ceil((this->getOutHeight() - MAX_THREAD_SIZE) \
							* 1.0f / MAX_THREAD_SIZE) + 1;
			_box_num_width = ceil((this->getOutWidth() - MAX_THREAD_SIZE) \
							* 1.0f / MAX_THREAD_SIZE) + 1;
			_box_in_height = (MAX_THREAD_SIZE - 1) * stride_height + filter_height;
			_box_in_width = (MAX_THREAD_SIZE - 1) * stride_width + filter_width;
			_box_out_height = MAX_THREAD_SIZE > _out_height \
					? _out_height : MAX_THREAD_SIZE;
			_box_out_width = MAX_THREAD_SIZE > _out_width \
					? _out_width : MAX_THREAD_SIZE;
			
			int pow2Length = _out_height; 
			if(pow2Length & (pow2Length - 1)){
				while(pow2Length & (pow2Length - 1)){
					pow2Length &= pow2Length - 1;
				}
				pow2Length *= 2;
			}
			_thread_height = pow2Length > MAX_THREAD_SIZE \
							 ? MAX_THREAD_SIZE : pow2Length;

			pow2Length = _out_width; 
			if(pow2Length & (pow2Length - 1)){
				while(pow2Length & (pow2Length - 1)){
					pow2Length &= pow2Length - 1;
				}
				pow2Length *= 2;
			}
			_thread_width = pow2Length > MAX_THREAD_SIZE \
							? MAX_THREAD_SIZE : pow2Length;

			_overlap_height = _filter_height - stride_height;
			_overlap_width = _filter_width - stride_width;

		}

    LocalConnectParam(LayerType layer_type, string name, \
		const int pad_height, const int pad_width, \
		const int stride_height, const int stride_width, \
		const int filter_height, const int filter_width, const int filter_channel, \
		LocalConnectParam* lc_par) \
		: _in_height(lc_par->getOutHeight()), _in_width(lc_par->getOutWidth()), \
		_stride_height(stride_height), _stride_width(stride_width), \
		_in_channel(lc_par->getOutChannel()), _pad_height(pad_height), \
		_filter_height(filter_height), _filter_width(filter_width) {

            this->_layer_type = layer_type;
			this->_name = name;
			if(filter_channel != 0)
				_out_channel = filter_channel;
			else
				_out_channel = _in_channel;

			this->type = PARAM_CONNECT_TYPE_LOCAL;

			_padded_in_height = _in_height + 2 * pad_height;
			_padded_in_width = _in_width + 2 * pad_height;
			_out_height = ceil(((_padded_in_height - filter_height)*1.0f) / stride_height) + 1;
			_out_width = ceil(((_padded_in_width - filter_width)*1.0f) / stride_width) + 1;
			_box_num_height = ceil((this->getOutHeight() - MAX_THREAD_SIZE) \
							* 1.0f / MAX_THREAD_SIZE) + 1;
			_box_num_width = ceil((this->getOutWidth() - MAX_THREAD_SIZE) \
							* 1.0f / MAX_THREAD_SIZE) + 1;
	
			_box_out_height = MAX_THREAD_SIZE > _out_height \
					? _out_height : MAX_THREAD_SIZE;
			_box_out_width = MAX_THREAD_SIZE > _out_width \
					? _out_width : MAX_THREAD_SIZE;

			_box_in_height = (MAX_THREAD_SIZE - 1) * stride_height + filter_height;
			_box_in_width = (MAX_THREAD_SIZE - 1) * stride_width + filter_width;
			
			int pow2Length = _out_height; 
			if(pow2Length & (pow2Length - 1)){
				while(pow2Length & (pow2Length - 1)){
					pow2Length &= pow2Length - 1;
				}
				pow2Length *= 2;
			}
			_thread_height = pow2Length > MAX_THREAD_SIZE \
							 ? MAX_THREAD_SIZE : pow2Length;

			pow2Length = _out_width; 
			if(pow2Length & (pow2Length - 1)){
				while(pow2Length & (pow2Length - 1)){
					pow2Length &= pow2Length - 1;
				}
				pow2Length *= 2;
			}
			_thread_width = pow2Length > MAX_THREAD_SIZE \
							? MAX_THREAD_SIZE : pow2Length;
			
			_overlap_height = _filter_height - stride_height;
			_overlap_width = _filter_width - stride_width;


		}

    inline int getInHeight() {
        return _in_height;
    }
    inline int getInWidth() {
        return _in_width;
    }
    inline int getInChannel() {
        return _in_channel;
    }
    inline int getOutHeight() {
        return _out_height;
    }
    inline int getOutWidth() {
        return _out_width;
    }
    inline int getFilterHeight() {
        return _filter_height;
    }
    inline int getFilterWidth() {
        return _filter_width;
    }
    inline int getOutChannel() {
        return _out_channel;
    }
    inline int getPaddedInHeight() {
        return _padded_in_height;
    }
    inline int getPaddedInWidth() {
        return _padded_in_width;
    }

    inline int getStrideHeight(){
        return _stride_height;
    }
    inline int getStrideWidth(){
        return _stride_width;
    }
    inline int getPadHeight(){
        return _pad_height;
    }
    inline int getPadWidth(){
        return _pad_width;
    }
	int getOverlapHeight(){
		return _overlap_height;
	}
	int getOverlapWidth(){
		return _overlap_width;
	}
	int getThreadHeight(){
		return _thread_height;
	}
	int getThreadWidth(){
		return _thread_width;
	}
    void printParam(){
        Param::printParam();
        cout << "\nin_height: " << _in_height \
			<< "\nin_width: " << _in_width \
				<< "\nin_channel: " << _in_channel \
                << "\nfilter_height: " << _filter_height \
                << "\nfilter_width: " << _filter_width \
				<< "\nfilter_channel: " << _out_channel \
				<< "\npad_height: " << _pad_height \
				<< "\npad_width: " << _pad_width \
				<< "\nstride_height: " << _stride_height \
				<< "\nstride_width: " << _stride_width;
    }
	inline int getBoxNumHeight(){
		return _box_num_height;
	}
	inline int getBoxNumWidth(){
		return _box_num_width;
	}
	inline int getBoxInHeight(){
		return _box_in_height;
	}
	inline int getBoxInWidth(){
		return _box_in_width;
	}
	inline int getBoxOutHeight(){
		return _box_out_height;
	}
	inline int getBoxOutWidth(){
		return _box_out_width;
	}

private:
    int _in_height;
	int _in_width;
    int _pad_height;
    int _pad_width;
    int _padded_in_height;
    int _padded_in_width;
    int _stride_height;
    int _stride_width;
    int _in_channel;
    int _filter_height; ///>在卷积中是filter，在pooling中是pool
    int _filter_width; ///>在卷积中是filter，在pooling中是pool
    int _out_height;
    int _out_width;
    int _out_channel;
	int _box_in_height; ///>用来计算一个box输出的
	int _box_in_width; ///>用来计算一个box输出的卷积输入
	int _box_out_height; 
	int _box_out_width; 
	int _box_num_height;  ///>总的box个数的行 
	int _box_num_width;  ///>总的box个数的列 
	int _thread_height;
	int _thread_width;
	int _overlap_height;
	int _overlap_width;
};

/// \brief 全连接层的参数，展开图片为一个矢量保存数据
///
/// 可以针对每一个值做某种操作，例如Relu、sigmoid、tanh等，
/// 此处不需要训练
class FullConnectParam : public virtual Param {
public:
    FullConnectParam() { }
    virtual ~FullConnectParam() { }
    FullConnectParam(LayerType layer_type, string name, \
        const int num_in, const int num_out) \
		: _num_in(num_in), _num_out(num_out) {
            this->_layer_type = layer_type;
			this->_name = name;
			this->type = PARAM_CONNECT_TYPE_FULL;
		}
    FullConnectParam(LayerType layer_type, string name, \
		const int num_out, Param* par){
            this->_layer_type = layer_type;
			this->_name = name;
			this->type = PARAM_CONNECT_TYPE_FULL;
			
			///由传递进来的层类型决定计算方式
			ConnectType ct = par->getConnectType();
			if(ct == PARAM_CONNECT_TYPE_LOCAL)
				_num_in = par->getOutHeight()*par->getOutWidth()*par->getOutChannel(); 
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
    void printParam(){
        Param::printParam();
        cout << "\nnum_in: " << _num_in \
				<< "\nnum_out: " << _num_out;
    }

private:
    int _num_in;
    int _num_out;
};

class ConvParam : public TrainParam, public LocalConnectParam {
public:
    ConvParam(){}

    ~ConvParam(){}

    ConvParam(const LayerType layer_type, const string name, \
            const float w_lr, \
			const float b_lr, const float momentum, \
			const float weight_decay, \
            const int in_height, const int in_width, \
            const int pad_height, const int pad_width, \
			const int stride_height, \
			const int stride_width, const int in_channel, \
            const int filter_height, const int filter_width, \
			const int filter_channel) \
            : TrainParam(w_lr, b_lr, momentum, weight_decay), \
              LocalConnectParam(layer_type, name, in_height, in_width, \
		            pad_height, pad_width, stride_height, stride_width, \
					in_channel, filter_height, \
					filter_width, filter_channel) {}

    ConvParam(const LayerType layer_type, const string name, const float w_lr, \
            const float b_lr, const float momentum, \
			const float weight_decay, const int pad_height, const int pad_width, \
            const int stride_height, const int stride_width, const int filter_height, \
			const int filter_width, \
            const int filter_channel, LocalConnectParam *lc_par) \
            : TrainParam(w_lr, b_lr, momentum, weight_decay), \
              LocalConnectParam(layer_type, name, pad_height, pad_width, stride_height, \
					  stride_width, \
		            filter_height, filter_width, filter_channel, lc_par)  {}
    void printParam(){
        LocalConnectParam::printParam();
        TrainParam::printParam();
    }
};

class PoolParam : public LocalConnectParam {
public:
		PoolParam() {}
		~PoolParam() {}

    	PoolParam(const LayerType layer_type, const string name, \
            const int in_height, const int in_width, \
			const int pad_height, const int pad_width, \
			const int stride_height, const int stride_width, \
			const int in_channel, const int filter_height, \
			const int filter_width, \
			const int filter_channel, PoolingType p_type) 
            :  LocalConnectParam(layer_type, name, in_height, in_width, \
					pad_height, pad_width, stride_height, stride_width, \
					in_channel, filter_height, \
					filter_width, filter_channel) , \
			_p_type(p_type) {}

    	PoolParam(const LayerType layer_type, const string name, \
            const int pad_height, const int pad_width, \
			const int stride_height, const int stride_width, \
			const int filter_height, const int filter_width, \
			const int filter_channel, \
			LocalConnectParam* lc_par, PoolingType p_type) 
            :  LocalConnectParam(layer_type, name, pad_height, \
					pad_width, stride_height, \
					stride_width, \
					filter_height, filter_width, \
					filter_channel, lc_par), _p_type(p_type){}

		inline PoolingType getPoolType(){
			return _p_type;
		}
        void printParam(){
            LocalConnectParam::printParam();
        }


private:
	PoolingType _p_type;	
};


/// \brief 可以进行训练的全连接层
class InnerParam : public TrainParam, public FullConnectParam {
public:
    InnerParam(){}

    ~InnerParam() {}

    InnerParam(const LayerType layer_type, const string name, \
		const float w_lr, const float b_lr, const float momentum, \
		const float weight_decay, \
		const int num_in, const int num_out) \
        : TrainParam(w_lr, b_lr, momentum, weight_decay),
          FullConnectParam(layer_type, name, num_in, num_out){}

    InnerParam(const LayerType layer_type, const string name, \
        const float w_lr, const float b_lr, \
        const float momentum, const float weight_decay, \
        const int num_out, Param* par) \
        : TrainParam(w_lr, b_lr, momentum, weight_decay),  \
          FullConnectParam(layer_type, name, num_out, par) {}
    void printParam(){
        FullConnectParam::printParam();
        TrainParam::printParam();
    }
};

#endif
