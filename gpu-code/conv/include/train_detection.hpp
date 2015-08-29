///
/// \file train_detection.hpp
/// \brief 
///


#ifndef TRAINDETECTION_H_
#define TRAINDETECTION_H_

#include "train_model.hpp"
#include "python_interface.hpp"


/// \brief
///
template<typename Dtype>
class TrainDetection : public TrainModel<Dtype> {
private:
	PythonInterface<Dtype> *_python_interface;

public:
    TrainDetection(const int master_pid, const int pid) \
		: TrainModel<Dtype>(master_pid, pid) {
			_python_interface = new PythonInterface<Dtype>();
		}
	~TrainDetection() {
		delete _python_interface;
	}

    void createPixelAndCoord();
	void parseImgBinary(int num_process);

	void createMPIDist();

	void forwardLastLayer();
	void backwardLastLayer();

    void train();
	void sendAndRecvForManager();

};

#include "../src/train_detection.cpp"

#endif
