///
///  \file conv3.cu
///

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include "train_classification.hpp"
#include "convnet.hpp"

using namespace std;

int Param::_minibatch_size = 0;

int main(int argc, char** argv){

	TrainClassification<float> *cifar_model = new TrainClassification<float>(true, false);

	cifar_model->parseNetJson("script/cifar10.json");
	cout << "done1\n";
	cifar_model->parseImgBinary("", "");
	cifar_model->createLayer();
	cifar_model->createWBias();
	cifar_model->createPixelAndLabel();
	cifar_model->createYDEDY();
	cifar_model->initWeightByRandom();
	cifar_model->train();
	 	
	delete cifar_model;


	return 0;
}

















