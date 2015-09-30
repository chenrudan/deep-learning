///
/// \file recommendation_layer.cu
/// @brief

#include "recommendation_layer.hpp"

using namespace std;

template <typename Dtype>
RecommendationLayer<Dtype>::RecommendationLayer(FullConnectParam* fcp){

	this->_fcp           = fcp;
	if(_fcp->getLayerType() == RECOMMENDCOMPATIBLE)
		_is_compatible = true;
	else
		_is_compatible = false;
}

template <typename Dtype>
RecommendationLayer<Dtype>::~RecommendationLayer() {
	delete[] y_CPU;
	delete[] x_CPU;
	delete[] dE_dx_CPU;
	if(_is_compatible){
		delete[] w_CPU;
		delete[] dE_dw_CPU;
	}
	delete[] h_labels;
}

template <typename Dtype>
void RecommendationLayer<Dtype>::initCuda() {

	y_CPU             = new Dtype[_fcp->getMinibatchSize()/2];
	x_CPU 				= new Dtype[_fcp->getMinibatchSize()*_fcp->getNumIn()];
	dE_dx_CPU 				= new Dtype[_fcp->getMinibatchSize()*_fcp->getNumIn()];
	if(_is_compatible){
		w_CPU				= new Dtype[_fcp->getNumIn()*_fcp->getNumOut()];
		dE_dw_CPU				= new Dtype[_fcp->getNumIn()*_fcp->getNumOut()];
		dE_dw				= new Matrix<Dtype>(_fcp->getNumIn(), _fcp->getNumOut());
		gaussRand(w_CPU, _fcp->getNumIn()*_fcp->getNumOut(), 0.1);
	}

	h_labels = new int[_fcp->getMinibatchSize()];

}

template <typename Dtype>
double RecommendationLayer<Dtype>::computeError(Matrix<Dtype>* x, \
		Matrix<int>* labels){ 
//	x->reValue(_fcp->getNumIn(), true);
//	x->showValue("data");


	x->copyToHost(x_CPU, x->getNumEles());
	labels->copyToHost(h_labels, labels->getNumEles());
	
	memset(y_CPU, 0, (_fcp->getMinibatchSize()/2)*sizeof(int));
	
	double result = 0;
	for(int i=0; i < _fcp->getMinibatchSize()/2; i++){
		int pos_or_neg = 1;
		if(h_labels[2*i] < 0)
			pos_or_neg = -1;

		if(!_is_compatible){
			for(int j=0; j < x->getNumCols(); j++){
				y_CPU[i] += pow(x_CPU[i*2*x->getNumCols() + j] \
						- x_CPU[(i*2+1)*x->getNumCols() + j], 2); 
			}	
		}else{

			for(int k=0; k < _fcp->getNumOut(); k++){
				Dtype ele = 0;
				for(int j=0; j < _fcp->getNumIn(); j++){
					ele += (x_CPU[i*2*x->getNumCols() + j] \
							- x_CPU[(i*2+1)*x->getNumCols() + j])\
						*w_CPU[j*_fcp->getNumOut()+k]; 
				}
				y_CPU[i] += pow(ele, 2);
			}
		}
	//	cout << h_labels[i] << ",   y_cpu: "<< y_CPU[i] << endl;

		/***用log来算
		if(y_CPU[i] != 0)
			result -= log(y_CPU[i]);
		***/
		//负样本因为要减去结果值
		result += pos_or_neg*y_CPU[i];

		cout << h_labels[1+i*2] << "\t" << h_labels[2*i] << "\t";
		cout << y_CPU[i] << "\n";
	}
//cout << result << endl;	
	return result;

}

template <typename Dtype>
void RecommendationLayer<Dtype>::computeDerivsOfInput(Matrix<Dtype>* dE_dx){
	
	if(_is_compatible)
		memset(dE_dw_CPU, 0, _fcp->getNumIn()*_fcp->getNumOut()*sizeof(Dtype));
	for(int i=0; i < _fcp->getMinibatchSize()/2; i++){
		int pos_or_neg = 1;
		if(h_labels[2*i] < 0)
			pos_or_neg = -1;

		if(!_is_compatible){
			for(int j=0; j < dE_dx->getNumCols(); j++){
				dE_dx_CPU[i*2*dE_dx->getNumCols()+j] \
						= (x_CPU[i*2*dE_dx->getNumCols() + j] \
							- x_CPU[(i*2+1)*dE_dx->getNumCols() + j]) \
						*2*x_CPU[i*2*dE_dx->getNumCols() + j];
				dE_dx_CPU[(i*2+1)*dE_dx->getNumCols()+j] \
						= (x_CPU[(i*2+1)*dE_dx->getNumCols() + j] \
							- x_CPU[i*2*dE_dx->getNumCols() + j]) \
						*2*x_CPU[(i*2+1)*dE_dx->getNumCols() + j];

				/***用log算的时候的求导
				if(y_CPU[i] < 0.00001){
					dE_dx_CPU[i*2*dE_dx->getNumCols()+j] = 0;
					dE_dx_CPU[(i*2+1)*dE_dx->getNumCols()+j] = 0;
				}else{
					dE_dx_CPU[i*2*dE_dx->getNumCols()+j] \
						= (x_CPU[(i*2+1)*dE_dx->getNumCols() + j] \
							- x_CPU[i*2*dE_dx->getNumCols() + j]) / pow(y_CPU[i],2);
					dE_dx_CPU[(i*2+1)*dE_dx->getNumCols()+j] \
						= (x_CPU[i*2*dE_dx->getNumCols() + j] \
							- x_CPU[(i*2+1)*dE_dx->getNumCols() + j]) / pow(y_CPU[i],2);
				}***/
			}
		}else{

			for(int j=0; j < _fcp->getNumIn(); j++){
				Dtype tmp = 0;
				for(int k=0; k < _fcp->getNumOut(); k++){
					tmp += pow(w_CPU[j*_fcp->getNumOut()+k], 2);
				}
				if(y_CPU[i] == 0.0f){
					dE_dx_CPU[i*2*dE_dx->getNumCols()+j] = 0;
					dE_dx_CPU[(i*2+1)*dE_dx->getNumCols()+j] = 0;
				}else{
					dE_dx_CPU[i*2*dE_dx->getNumCols()+j] \
						= pos_or_neg*(x_CPU[i*2*dE_dx->getNumCols() + j] \
							- x_CPU[(i*2+1)*dE_dx->getNumCols() + j])*tmp;
					dE_dx_CPU[(i*2+1)*dE_dx->getNumCols()+j] \
						= pos_or_neg*(x_CPU[(i*2+1)*dE_dx->getNumCols() + j] \
							- x_CPU[i*2*dE_dx->getNumCols()+j])*tmp;
				}
			}
		}

		if(_is_compatible){
			
			for(int k=0; k < _fcp->getNumOut(); k++){
				for(int j=0; j < _fcp->getNumIn(); j++){
					if(y_CPU[i] == 0.0f){
						dE_dw_CPU[j*_fcp->getNumOut()+k] += 0;
					}else{
						dE_dw_CPU[j*_fcp->getNumOut()+k] += pos_or_neg \
							*pow(x_CPU[(i*2+1)*dE_dx->getNumCols() + j] \
							- x_CPU[i*2*dE_dx->getNumCols() + j], 2) \
							*w_CPU[j*_fcp->getNumOut()+k];
					}
				}
			}
		}
	}
//	cout << "-------------dedw_cpu-----------\n";
	if(_is_compatible){
		for(int k=0; k < _fcp->getNumOut(); k++){
			for(int j=0; j < _fcp->getNumIn(); j++){
//			cout << w_CPU[j*_fcp->getNumOut()+k] << ":";
				w_CPU[j*_fcp->getNumOut()+k] -= 0.00005*dE_dw_CPU[j*_fcp->getNumOut()+k]/_fcp->getMinibatchSize();
//			cout << w_CPU[j*_fcp->getNumOut()+k] << "\t";
			}
//		cout << endl;
		}
	}

	dE_dx->copyFromHost(dE_dx_CPU, dE_dx->getNumEles());


//dE_dx->showValue("Recommendation_dedx");

}


