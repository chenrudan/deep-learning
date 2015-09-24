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
protected:
    ModelComponent<Dtype> *_model_component;
    LoadLayer<Dtype> *_load_layer;
    float _likelihood;    ///>cost function的输出值
	int _error;    ///>分类的error个数
	//early stopping
	float _min_likelihood;       ///>early stopping所控制得到的最小cost
	vector<float> _strip_likelihood;  ///>用来控制early stopping
	int _min_epoch;
	int _min_error;
	int _num_strip;
	bool _is_stop;   ///>训练是否由于early stopping而中断
	bool _has_valid;
	bool _is_test;
	int _num_data_type;  //train是0，valid是1，test是2

public:
    TrainModel(const int master_pid, const int pid, bool has_valid, bool is_test);
    virtual ~TrainModel();

    void parseImgBinary(int num_process);
    void parseNetJson(string json_file);

    void createLayerForWorker();
    void createYDEDYForWorker();
    void createWBiasForManager();
    void createWBiasForWorker();
	
	virtual void createMPIDist();
	virtual void createDataMPIDist(int multi) {}

    void initWeightAndBcastByRandom();
    void initWeightAndBcastByFile(vector<string> w_file, vector<string> bias_file);
    void forwardPropagate();
    void backwardPropagate();
    void computeAndUpdatePars();
	void sendAndRecvWBiasForWorker(const int epoch_idx, \
			const int batch_idx, int flag);

	virtual void forwardLastLayer() {}
	virtual void backwardLastLayer() {}

    virtual void train() {}

	//返回是true就停下，返回是false就继续执行
	void earlyStopping(int epoch_idx);

};

#include "../src/train_model.cpp"

#endif
