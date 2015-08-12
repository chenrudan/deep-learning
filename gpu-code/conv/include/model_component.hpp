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
    int _num_need_train_layers;
    int _img_size;
    int _img_channel;
    int _one_img_len;  ///>输入的一张图片的长度
	int _in_len_each_process;
    int _rank;  ///>进程id
    vector<Layer*> _layers;    ///>保存每个层的指针
    vector<Layer*> _layers_needed_train;
    vector<Param*> _layers_param;  ///>保存每一层的参数
    vector<Param*> _layers_need_train_param;
    vector<int> _w_len;   ///>需要训练的层的权重长度，用来进程间传递数据
    vector<int> _bias_len;
    vector<float> _w_init_gauss;
    vector< Matrix<Dtype>* > _w_for_manager; ///>保存需要训练层的权重指针
    vector< Matrix<Dtype>* > _bias_for_manager;

    vector< Matrix<Dtype>* > _y_for_worker;
    vector< Matrix<Dtype>* > _dE_dy_for_worker;
    vector< Matrix<Dtype>* > _y_for_compute_par;

    Matrix<Dtype>* _mini_train_data;  ///> 需要保存两组数据，一组当前作为运算，一组在运算的时候交换，下一次再计算
    Matrix<Dtype>* _mini_valid_data;
    Matrix< vector<int> > *_mini_train_label_for_voc;
    Matrix< vector<int> > *_mini_valid_label_for_voc;
    Matrix<Dtype>* _mini_data;
    Matrix< vector<int> > *_mini_label_for_voc;

    map<string, LayerType> _string_map_layertype;

public:

    ModelComponent();
    ~ModelComponent();

    friend class ModelTrain;

    void setImgSize(const int img_size){
        _img_size = img_size;
    }
    void setRank(const int rank){
        _rank = rank;
    }
    void setImgChannel(const int img_channel){
        _img_channel = img_channel;
    }
    void setOneImgLen(const int one_img_len){
        _one_img_len = one_img_len;
    }
	void setInLenEachProcess(const int in_len_each_process){
		_in_len_each_process = in_len_each_process;
	}
    void setNumLayers(const int num_layers){
        _num_layers = num_layers;
    }
    void setNumNeedTrainLayers(const int num_need_train_layers){
        _num_need_train_layers= num_need_train_layers;
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
        _num_train_batch = _num_train / (_minibatch_size * (_num_process - 1));
    }
    void setNumValidbatch(){
        _num_valide_batch = _num_valid / (_minibatch_size * (_num_process - 1));
    }
    void setEpoch(const int num_epoch){
        _num_epoch = num_epoch;
    }
    void setLayers(Layer<Dtype>* layer){
        _layers.push_back(layer);
    }
    void setNeedTrainLayers(Layer<Dtype>* need_train_layer){
        _layers.push_back(need_train_layer);
    }
    void setLayersParam(Param* param){
        _layers_param.push_back(param);
    }
    void setNeedTrainLayersParam(Param* param){
        _layers_need_train_param.push_back(param);
    }
    void setWLen(int w_len){
        _w_len.push_back(w_len);
    }
    void setBiasLen(int bias_len){
        _bias_len.push_back(bias_len);
    }
    void setW(Matrix<Dtype> *w){
        _w.push_back(w);
    }
    void setBias(Matrix<Dtype> *bias){
        _bias.push_back(bias);
    }
    void setY(Matrix<Dtype> *y){
        _y.push_back(y);
    }
    void setDEDY(Matrix<Dtype> *dE_dy) {
        _dE_dy.push_back(dE_dy);
    }

    int getRank(){
        return _rank;
    }
    int getImgSize(){
        return _img_size;
    }
    int getImgChannel(){
        return _img_channel;
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
    int getNumNeedTrainLayers(){
        return _num_need_train_layers;
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
    vector< Layer<Dtype>* > getLayers(){
        return _layers;
    }
    vector< Layer<Dtype>* > getNeedTrainLayers(){
        return _layers_needed_train;
    }
    vector<Param*> getLayersParam(){
        return _layers_param;
    }
    vector<Param*> getNeedTrainLayersParam(){
        return _layers_need_train_param;
    }
    vector<int> getWLen(){
        return _w_len;
    }
    vector<int> getBiasLen(){
        return _bias_len;
    }
    vector< Matrix<Dtype>* > getW(){
        return _w;
    }
    vector< Matrix<Dtype>* > getBias(){
        return _bias;
    }
    vector< Matrix<Dtype>* > getY(){
        return _y;
    }
    vector< Matrix<Dtype>* > getDEDY(){
        return _dE_dy;
    }

};

#include "../src/model_component.cpp"

#endif
