///
/// \file model_component.cpp
/// @brief
#include "model_component.hpp"

using namespace std;

template <typename Dtype>
ModelComponent<Dtype>::ModelComponent() {
	_string_map_layertype["CONVOLUTION"] = CONVOLUTION;
	_string_map_layertype["POOLING"] = POOLING;
	_string_map_layertype["SIGMOID"] = SIGMOID;
	_string_map_layertype["RECTIFIED"] = RECTIFIED;
	_string_map_layertype["INNERPRODUCT"] = INNERPRODUCT;
	_string_map_layertype["SOFTMAX"] = SOFTMAX;
	_string_map_layertype["DROPOUT"] = DROPOUT;

	_string_map_pooltype["MAX_POOLING"] = MAX_POOLING;
	_string_map_pooltype["AVG_POOLING"] = AVG_POOLING;


	_num_need_train_layers = 0;
}


