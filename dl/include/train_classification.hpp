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
    TrainClassification(bool has_valid, bool is_test) \
		: TrainModel<Dtype>(has_valid, is_test) {}
    ~TrainClassification() {}

    void createPixelAndLabel();
	void parseImgBinary(string train_file, string valid_file);

	void forwardLastLayer();
	void backwardLastLayer();
	virtual void train();
	virtual void test() {}

};

#include "../src/train_classification.cpp"

#endif
