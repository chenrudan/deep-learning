///
/// \file train_Recommendation.hpp
/// \brief 
///


#ifndef TRAINRecommendation_H_
#define TRAINRecommendation_H_

#include "train_classification.hpp"

/// \brief
///
template<typename Dtype>
class TrainRecommendation : public TrainClassification<Dtype> {
private:

public:
    TrainRecommendation(const int master_pid, const int pid) \
		: TrainClassification<Dtype>(master_pid, pid) {}
    ~TrainRecommendation() {}

	void parseImgBinary(int num_process, string train_file, string valid_file);

	void forwardLastLayer();
	void backwardLastLayer();
	void train();

};

#include "../src/train_recommendation.cpp"

#endif
