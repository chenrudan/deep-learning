///
/// \file train_classification.hpp
/// \brief 
///


#ifndef TRAINCLASSIFICATION_H_
#define TRAINCLASSIFICATION_H_

#include "train_model.hpp"

/// \brief
///
template<typename Dtype>
class TrainClassification : public TrainModel<Dtype> {
private:

public:
    TrainClassification(const int master_pid, const int pid) \
		: TrainModel<Dtype>(master_pid, pid) {}
    ~TrainClassification() {}

    void createPixelAndLabel();
	void parseImgBinary(int num_process);

	void forwardLastLayer();
	void backwardLastLayer();
	void createMPIDist();
	void train();
	void sendAndRecvForManager();

};

#include "../src/train_classification.cpp"

#endif
