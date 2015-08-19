///
/// \file train_model.hpp
/// \brief 
///


#ifndef TRAINMODEL_H_
#define TRAINMODEL_H_

#include "model_component.hpp"
#include "load_layer.hpp"
#include "mpi_distribute.hpp"

using namespace std;

/// \brief
///
template<typename Dtype>
class TrainModel {
private:
    ModelComponent<Dtype> *_model_component;
    LoadCifar10<Dtype> *_voc;
    float _likelihood;
	int _error;
    int _cur_batch_idx; 
    map<string, int> transmit_data_id;

public:
    TrainModel(const int pid);
    ~TrainModel();

    void initModel(int num_process, string json_file);
    void parseImgBinary(int num_process);
    void parseNetJson(string json_file);
    void createPixelAndLabel();
    void createLabelNum();

    void createLayerForWorker();
    void createYDEDYForWorker();
    void createWBiasForManager();
    void createWBiasForWorker();
	void createMPIDist();

    void initWeightAndBcast();
    float forwardPropagate();
    void backwardPropagate();
    void computeAndUpdatePars();

    void train();
    void valid();

	void sendAndRecvForManager();

};

#include "../src/train_model.cpp"

#endif
