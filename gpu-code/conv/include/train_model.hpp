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
    LoadVOC<Dtype> *_voc;
    float _likelihood;
	int _error;
    int _cur_batch_idx; 
    map<string, int> transmit_data_id;
	//early stopping
	float _min_likelihood;
	vector<float> _strip_likelihood;
	int _min_epoch;
	int _min_error;
	int _num_strip;
	bool _is_stop;

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
    void forwardPropagate();
    void backwardPropagate();
    void computeAndUpdatePars();

	void backwardClassification();
	void backwardDetection();
	void forwardClassification();
	void forwardDetection();

	void trainClassification();
	void trainDetection();

    void train();
    void valid();

	//返回是true就停下，返回是false就继续执行
	void earlyStopping(int epoch_idx);
	void sendAndRecvForManager();

};

#include "../src/train_model.cpp"

#endif
