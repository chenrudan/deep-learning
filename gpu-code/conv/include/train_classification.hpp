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
    TrainClassification(const int master_pid, const int pid, bool has_valid, bool is_test) \
		: TrainModel<Dtype>(master_pid, pid, has_valid, is_test) {}
    ~TrainClassification() {}

    void createPixelAndLabel();
	void parseImgBinary(int num_process, string train_file, string valid_file);

	void forwardLastLayer();
	void backwardLastLayer();
	void createMPIDist();
	void createDataMPIDist(int multi);
	virtual void train();
	virtual void test() {}
	void sendAndRecvForManager();

};

#include "../src/train_classification.cpp"

#endif
