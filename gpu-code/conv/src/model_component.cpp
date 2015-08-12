///
/// \file model_component.cpp
/// @brief

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

	_num_need_train_layers = 0;
	_num_local_layers = 0;
	_num_need_train_local_layers = 0;
}

template <typename Dtype>
ModelComponent<Dtype>::~ModelComponent() {
	for (int i = 0; i < _num_layers; ++i) {
		Param *param = _layers_param.pop_back();
		delete param;
		if (_rank > 0) {
			Layer *layer = _layers.pop_back();
			delete layer;
		}
	}
	//主控制进程new过w和bias
	if (_rank == 0) {
		for (int i = 0; i < _num_need_train_layers; ++i){
			Matrix<Dtype> tmp = _w.pop_back();
			delete tmp;
			tmp = _bias.pop_back();
			delete tmp;
		}
	}
}


