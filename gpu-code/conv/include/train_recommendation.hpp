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
    TrainRecommendation(const int master_pid, const int pid, bool has_valid = true, bool is_test = false) \
		: TrainClassification<Dtype>(master_pid, pid, has_valid, is_test) {}
    ~TrainRecommendation() {}

	void parseImgBinary(int num_process, string train_file, string train_matches, string test_file);

	void forwardLastLayer();
	void backwardLastLayer();
	void train();
	void test();

};

#include "../src/train_recommendation.cpp"

#endif
