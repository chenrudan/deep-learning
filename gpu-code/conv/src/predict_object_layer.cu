///
/// \file predict_object_layer.cu
/// @brief

#include "predict_object_layer.hpp"

using namespace std;

template <typename Dtype>
PredictObjectLayer<Dtype>::PredictObjectLayer<Dtype>(FullConnectParam* fcp){
	_fcp = fcp;
}

template <typename Dtype>
PredictObjectLayer<Dtype>::~PredictObjectLayer<Dtype>() {
	delete[] _h_x;
	delete[] _h_coord;
	delete[] _h_dE_dx;
}

template <typename Dtype>
void PredictObjectLayer<Dtype>::initCuda() {
	_h_x            = new Dtype[_fcp->getMinibatchSize()*_fcp->getNumOut()];
	_h_coord		= new int[_fcp->getMinibatchSize()*_fcp->getNumOut()];
	_h_dE_dx		= new Dtype[_fcp->getMinibatchSize()*_fcp->getNumOut()];
}

template <typename Dtype>
double PredictObjectLayer<Dtype>::computeError(Matrix<Dtype> *x, \
		Matrix<int>* coord){
	x->copyToHost(_h_x, x->getNumEles());
	coord->copyToHost(_h_coord, coord->getNumEles());
	double error = 0;
	for(int i = 0; i < _fcp->getMinibatchSize()*4*MAX_OBJECT_NUM; i++){
		error += (_h_x[i] - (double)_h_coord[i]/500.0) \
				 *(_h_x[i] - (double)_h_coord[i]/500.0);
	}
//	x->showValue("x");
//	coord->showValue("coord");
//	coord->showValue("coord");
//	cout << error << endl;
	return error /= 2;
}


template <typename Dtype>
void PredictObjectLayer<Dtype>::computeDerivsOfInput(Matrix<Dtype>* dE_dx){
	memset(_h_dE_dx, 0, dE_dx->getNumEles()*sizeof(Dtype));
	for(int i = 0; i < _fcp->getMinibatchSize()*MAX_OBJECT_NUM*4; i++){
		_h_dE_dx[i] = _h_x[i] - (double)_h_coord[i]/500.0;
	}
	dE_dx->copyFromHost(_h_dE_dx, dE_dx->getNumEles());
//	dE_dx->showValue("predict dE_dx");
}

