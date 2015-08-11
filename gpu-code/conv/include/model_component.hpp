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

    int _num_process;
    int _num_train; ///>model的参数
    int _num_valid;
	int _num_train_each_process;
	int _num_valid_each_process;
    int _minibatch_size;
    int _num_train_batch;
    int _num_valid_batch;
    int _num_epoch;
    int _num_layers;
    int _num_local_layers;
    int _num_need_train_layers;
    int _num_need_train_local_layers;
    int _one_img_len;  ///>输入的一张图片的长度
	int _in_len_each_process;
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
friend class ModelTrain;

public:

    ModelComponent() {}
    ~ModelComponent() {}

    void setOneImgLen(const int one_img_len){
        _one_img_len = one_img_len;
    }
	void setInLenEachProcess(const int in_len_each_process){
		_in_len_each_process = in_len_each_process;
	}
    void setNumLayers(const int num_layers){
        _num_layers = num_layers;
    }
    void setNumLocalLayers(const int num_local_layers){
        _num_local_layers = num_local_layers;
    }
    void setNumNeedTrainLayers(const int num_need_train_layers){
        _num_need_train_layers= num_need_train_layers;
    }
    void setNumNeedTrainLocalLayers(const int num_need_train_local_layers){
        _num_need_train_local_layers= num_need_train_local_layers;
    }
    void setNumProcess(const int num_process){
        _num_process = num_process;
    }
    void setNumTrain(const int num_train){
        _num_train = num_train;
    }
    void setNumValid(const int num_valid){
        _num_Valid = num_valid;
    }
    void setNumTrainEachProcess(const int num_train_each_process){
        _num_train_each_process = num_train_each_process;
    }
    void setNumValid(const int num_valid_each_process){
        _num_valid_each_process = num_valid_each_process;
    }
    void setMinibatchSize(const int minibatch_size){
        _minibatch_size = minibatch_size;
    }
    void setNumTrainbatch(){
        _num_batch = _num_train / (_minibatch_size * (_num_process - 1));
    }
    void setNumValidbatch(){
        _num_batch = _num_valid / (_minibatch_size * (_num_process - 1));
    }
    void setEpoch(const int num_epoch){
        _num_epoch = num_epoch;
    }
    void setLayers(Layer* layer){
        _layers_ptr.push_back(layer);
    }
    void setNeedTrainLayers(Layer* need_train_layer){
        _layers_ptr.push_back(need_train_layer);
    }
    void setLayersParam(Param* param){
        _layers_param_ptr.push_back(param);
    }
    void setNeedTrainLayersParam(Param* param){
        _layers_need_train_param_ptr.push_back(param);
    }
    void setWLen(int w_len){
        _w_len.push_back(w_len);
    }
    void setBiasLen(int bias_len){
        _bias_len.push_back(bias_len);
    }
    void setWPtr(Matrix<Dtype> *w){
        _w_ptr.push_back(w);
    }
    void setBiasPtr(Matrix<Dtype> *bias){
        _bias_ptr.push_back(bias);
    }
    void setYPtr(Matrix<Dtype> *y){
        _y_ptr.push_back(y);
    }
    void setDEDYPtr(Matrix<Dtype> *dE_dy){
        _dE_dy_ptr.push_back(dE_dy);
    }

    int getOneImgLen(){
        return _one_img_len;
    }
	int getInLenEachProcess(){
		return _in_len_each_process;
	}
    int getNumLayers(){
        return _num_layers;
    }
    int getNumLocalLayers(){
        return _num_local_layers;
    }
    int getNumNeedTrainLayers(){
        return _num_need_train_layers;
    }
    int getNumNeedTrainLocalLayers(){
        return _num_need_train_local_layers;
    }
    int getNumProcess(){
        return _num_process;
    }
    int getNumTrain(){
        return _num_train;
    }
    int getNumValid(){
        return _num_Valid;
    }
    int getNumTrainiEachProcess(){
        return _num_train_each_process;
    }
    int getNumValidEachProcess(){
        return _num_valid_each_process;
    }
    int getMinibatchSize(){
        return _minibatch_size;
    }
    int getNumTrainbatch(){
        return _num_train_batch;
    }
    int getNumValidbatch(){
        return _num_valid_batch;
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
    vector<Param*> getNeedTrainLayersParam(){
        return _layers_need_train_param_ptr;
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
