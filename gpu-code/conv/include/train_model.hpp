///
/// \file train_model.hpp
/// \brief 
///


#ifndef TRAINMODEL_H_
#define TRAINMODEL_H_

#include "model_component.hpp"
#include "load_layer.hpp"
#include "inner_product_layer.hpp"
#include "logistic.hpp"
#include "sigmoid_layer.hpp"
#include "relu_layer.hpp"
#include "convnet.hpp"
#include "pooling_layer.hpp"
#include "dropout_layer.hpp"

using namespace std;

/// \brief
///
template<typename Dtype>
class TrainModel {
private:
    ModelComponent<Dtype> *_model_component;
    LoadVOC<Dtype> *_voc;
    float _likelihood;
	int _error;
    int _cur_batch_idx; 
    map<string, int> transmit_data_id;


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
    void createWBiasForWorker();

    void initWeightAndBcast();
    float forwardPropagate();
    void backwardPropagate();
    void updatePars();

    void train();
    void valid();

};

#include "../src/train_model.cpp"

#endif
