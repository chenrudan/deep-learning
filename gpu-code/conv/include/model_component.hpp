///
/// \file model_component.hpp
/// \brief 继承数据类，拥有矩阵的特性
///


#ifndef MODELCOMPONENT_H_
#define MODELCOMPONENT_H_

#include <vector>
#include <map>
#include "matrix.hpp"
#include "param.h"
#include "layer.hpp"
#include "mpi_distribute.hpp"

using namespace std;


/// \brief 实现了网络的组件，例如由几个卷积层几个全链接层构成
///
template<typename Dtype>
class ModelComponent {

template<typename D>
friend class TrainModel;

template<typename E>
friend class TrainDetection;

template<typename F>
friend class TrainClassification;

template<typename G>
friend class TrainRecommendation;

private:

    int _num_process;
    long long _num_train; ///>model的参数
    int _num_valid;
	long long _num_train_each_process;
	int _num_valid_each_process;
    int _minibatch_size;
    int _num_train_batch;
    int _num_valid_batch;
    int _num_epoch;
    int _num_layers;
    int _num_need_train_layers;
    int _img_height;
    int _img_width;
    int _img_channel;
    int _one_img_len;  ///>输入的一张图片的长度
	int _in_len_each_process;
	int _master_pid;  ///> 与之交换数据的主进程id号
    int _pid;  ///>进程id
	int _n_push;
	int _n_fetch;

    vector< Layer<Dtype>* > _layers;    ///>保存每个层的指针
    vector< Layer<Dtype>* > _layers_needed_train;
    vector<Param*> _layers_param;  ///>保存每一层的参数
    vector<Param*> _layers_need_train_param;
    vector<int> _w_len;   ///>需要训练的层的权重长度，用来进程间传递数据
    vector<int> _bias_len;
    vector<float> _w_init_gauss;
    vector< Matrix<Dtype>* > _w; ///>保存需要训练层的权重指针
    vector< Matrix<Dtype>* > _bias;

    vector< Matrix<Dtype>* > _y_for_worker;
    vector< Matrix<Dtype>* > _dE_dy_for_worker;
    vector< Matrix<Dtype>* > _y_needed_train;

    vector< Matrix<Dtype>* > _mini_data;  ///> 保存像素值
    vector< Matrix<int>* > _mini_label;   ///> 保存物体分类的类别
    vector< Matrix<int>* > _mini_coord;   ///> 保存物体识别的训练集坐标
	vector< Matrix<int>* > _mini_coord_num; ///>表示voc的每一张图中有几个label
	vector< Matrix<int>* > _mini_coord_label; ///>表示voc的每一张图中每一个坐标对应的label

	Matrix<int> *_mini_label_for_compute; 
	Matrix<int> *_mini_coord_for_compute;

    map<string, LayerType> _string_map_layertype;
	map<string, PoolingType> _string_map_pooltype;

	vector< MPIDistribute<Dtype>* > _send_recv_pixel;   ///>这是在每个detection或者classification进程之间传递数据
	vector< MPIDistribute<int>* > _send_recv_label;
	vector< MPIDistribute<Dtype>* > _send_recv_w;
	vector< MPIDistribute<Dtype>* > _send_recv_bias;
	
	vector< MPIDistribute<Dtype>* > _send_recv_pixel_between_task;  ///>这是从detection传给classification
	vector< MPIDistribute<int>* > _send_recv_label_between_task;

public:

    ModelComponent();
    ~ModelComponent();


    void setImgHeight(const int img_height){
        _img_height = img_height;
    }
    void setImgWidth(const int img_width){
        _img_width = img_width;
    }
    void setPid(const int pid){
        _pid = pid;
    }
    void setNPush(const int n_push){
        _n_push = n_push;
    }
    void setNFetch(const int n_fetch){
        _n_fetch = n_fetch;
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
    void setNumTrain(const long long num_train){
        _num_train = num_train;
    }
    void setNumValid(const int num_valid){
        _num_valid = num_valid;
    }
    void setNumTrainEachProcess(const long long num_train_each_process){
        _num_train_each_process = num_train_each_process;
    }
    void setNumValidEachProcess(const int num_valid_each_process){
        _num_valid_each_process = num_valid_each_process;
    }
    void setMinibatchSize(const int minibatch_size){
        _minibatch_size = minibatch_size;
    }
    void setNumTrainBatch(){
        _num_train_batch = _num_train / (_minibatch_size * (_num_process - 1));
    }
    void setNumValidBatch(){
        _num_valid_batch = _num_valid / (_minibatch_size * (_num_process - 1));
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
        _y_for_worker.push_back(y);
    }
    void setDEDY(Matrix<Dtype> *dE_dy) {
        _dE_dy_for_worker.push_back(dE_dy);
    }

    int getPid(){
        return _pid;
    }
    int getNPush(){
        return _n_push;
    }
    int getNFetch(){
        return _n_fetch;
    }
    int getImgHeight(){
        return _img_height;
    }
    int getImgWidth(){
        return _img_width;
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
    long long getNumTrain(){
        return _num_train;
    }
    int getNumValid(){
        return _num_valid;
    }
    long long getNumTrainiEachProcess(){
        return _num_train_each_process;
    }
    int getNumValidEachProcess(){
        return _num_valid_each_process;
    }
    int getMinibatchSize(){
        return _minibatch_size;
    }
    int getNumTrainBatch(){
        return _num_train_batch;
    }
    int getNumValidBatch(){
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
        return _y_for_worker;
    }
    vector< Matrix<Dtype>* > getDEDY(){
        return _dE_dy_for_worker;
    }

};

#include "../src/model_component.cpp"

#endif
