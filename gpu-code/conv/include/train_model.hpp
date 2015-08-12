///
/// \file train_model.hpp
/// \brief 继承数据类，拥有矩阵的特性
///


#ifndef TRAINMODEL_H_
#define TRAINMODEL_H_

#include "model_component.hpp"
#include "load_layer.hpp"
#include "inner_product_layer.hpp"
#include "logistic.hpp"
#include "param.h"
#include "matrix.hpp"
#include "sigmoid_layer.hpp"
#include "relu_layer.hpp"
#include "convnet.hpp"
#include "pooling_layer.hpp"
#include "dropout_layer.hpp"

using namespace std;

/// \brief 实现了网络在训练过程中会执行的一些操作
///
template<typename Dtype>
class TrainModel {
private:
    ModelComponent<Dtype> *_model_component;
    LoadVOC<Dtype> *_voc;

public:
    TrainModel();
    ~TrainModel();

    void initModel(int num_process, string json_file);
    void parseImgBinary(int num_process);
    void parseNetJson(string json_file);
    void createVOCPixelAndLabel();

    void createLayerForWorker();
    void createYDEDYForWorker();
    void createWBiasForManager();

    void initWeightAndBcast();
    float forwardPropagate();
    void backwardPropagate();
    void updatePars();

    void train();
    void valid();





};

#include "../src/matrix.cu"

#endif
