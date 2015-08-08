///
/// \file model_component.hpp
/// \brief 继承数据类，拥有矩阵的特性
///


#ifndef MODELCOMPONENT_H_
#define MODELCOMPONENT_H_

#include <vector>
#include "matrix.hpp"
#include "param.h"

using namespace std;

/// \brief 实现了网络的组件，例如由几个卷积层几个全链接层构成
///
template<typename Dtype>
class ModelComponent {
private:

    int _num_train; ///>model的参数
    int _num_Valid;
    int _num_minibatch;
    int _num_epoch;
    vector<Layer*> _layers_ptr;    ///>保存每个层的指针
    vector<Layer*> _layers_needed_train_ptr;
    vector<Param*> _layers_param_ptr;  ///>保存每一层的参数
    vector<Param*> _layers_need_train_param_ptr;
    vector<int> _w_len;   ///>需要训练的层的权重长度，用来进程间传递数据
    vector<int> _bias_len;
    vector< Matrix<Dtype>* > _w_ptr; ///>保存需要训练层的权重指针
    vector< Matrix<Dtype>* > _bias_ptr;
    vector< Matrix<Dtype>* > _y_ptr;
    vector< Matrix<Dtype>* > _dE_dy_ptr;

public:

    ModelComponent() {}
    ~ModelComponent() {}

    int getNumTrain(){
        return _num_train;
    }
    int getNumValid(){
        return _num_Valid;
    }
    int getNumMinibatch(){
        return _num_minibatch;
    }
    int getNumEpoch(){
        return _num_epoch;
    }
    vector<Layer*> getLayersPtr(){
        return _layers_ptr;
    }
    vector<Layer*> getNeedTrainLayersPtr(){
        return _layers_needed_train_ptr;
    }
    vector<Param*> getLayersParamPtr(){
        return _layers_param_ptr;
    }
    vector<int> getWLen(){
        return _w_len;
    }
    vector<int> getBiasLen(){
        return _bias_len;
    }
    vector< Matrix<Dtype>* > getWPtr(){
        return _w_ptr;
    }
    vector< Matrix<Dtype>* > getBiasPtr(){
        return _bias_ptr;
    }
    vector< Matrix<Dtype>* > getYPtr(){
        return _y_ptr;
    }
    vector< Matrix<Dtype>* > getDEDYPtr(){
        return _dE_dy_ptr;
    }

};

#include "../src/model_component.cpp"

#endif
