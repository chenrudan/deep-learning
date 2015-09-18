///
/// \file recommendation_layer.cu
/// @brief

#include "recommendation_layer.hpp"

using namespace std;

template <typename Dtype>
RecommendationLayer<Dtype>::RecommendationLayer(FullConnectParam* fcp){

	this->_fcp           = fcp;
}

template <typename Dtype>
RecommendationLayer<Dtype>::~RecommendationLayer() {
	delete[] y_CPU;
	delete[] x_CPU;
	delete[] dE_dx_CPU;
	delete[] w_CPU;
	delete[] dE_dw_CPU;
	delete[] h_labels;
}

template <typename Dtype>
void RecommendationLayer<Dtype>::initCuda() {

	y_CPU             = new Dtype[_fcp->getMinibatchSize()/2];
	x_CPU 				= new Dtype[_fcp->getMinibatchSize()*_fcp->getNumIn()];
	dE_dx_CPU 				= new Dtype[_fcp->getMinibatchSize()*_fcp->getNumIn()];
	w_CPU				= new Dtype[_fcp->getNumIn()*_fcp->getNumOut()];
	dE_dw_CPU				= new Dtype[_fcp->getNumIn()*_fcp->getNumOut()];

	h_labels = new int[_fcp->getMinibatchSize()];

//	gaussRand(w_CPU, _fcp->getNumIn()*_fcp->getNumOut(), 0.01);
	for(int i=0; i < _fcp->getNumIn()*_fcp->getNumOut(); i++){
		w_CPU[i] = 1;
	}
}

template <typename Dtype>
double RecommendationLayer<Dtype>::computeError(Matrix<Dtype>* x, \
		Matrix<int>* labels){ 
	x->reValue(_fcp->getNumIn(), true);
	x->showValue("data");

	x->copyToHost(x_CPU, x->getNumEles());
	labels->copyToHost(h_labels, labels->getNumEles());
	
	memset(y_CPU, 0, _fcp->getMinibatchSize()/2);
	for(int i=0; i < _fcp->getNumIn()*_fcp->getNumOut(); i++){
		w_CPU[i] = 1;
	}
	
	double result = 0;
	for(int i=0; i < _fcp->getMinibatchSize()/2; i++){
		if(h_labels[i] == 0){
			for(int j=0; j < x->getNumCols(); j++){
				y_CPU[i] += pow(x_CPU[i*2*x->getNumCols() + j] \
						- x_CPU[(i*2+1)*x->getNumCols() + j], 2); 
			}	
		}else if(h_labels[i] == 1){
			for(int k=0; k < _fcp->getNumOut(); k++){
				Dtype ele = 0;
				for(int j=0; j < _fcp->getNumIn(); j++){
					ele += (x_CPU[i*2*x->getNumCols() + j] \
							- x_CPU[(i*2+1)*x->getNumCols() + j])\
						*w_CPU[j*_fcp->getNumOut()+k]; 
				}
				y_CPU[i] += pow(ele, 2);
			}
		}else{
			cout << "match label not correct\n";
		}
		cout << "y_cpu: "<< y_CPU[i] << endl;
		//result -= log(y_CPU[i]);
	}
	cout << result << endl;
	cout << "\n";
	
	return result;

}

template <typename Dtype>
void RecommendationLayer<Dtype>::computeDerivsOfInput(Matrix<Dtype>* dE_dx){
	
	memset(dE_dw_CPU, 0, _fcp->getNumIn()*_fcp->getNumOut());
	for(int i=0; i < _fcp->getMinibatchSize()/2; i++){
		if(h_labels[i] == 0){
			for(int j=0; j < dE_dx->getNumCols(); j++){
				dE_dx_CPU[i*2*dE_dx->getNumCols()+j] \
					= (x_CPU[(i*2+1)*dE_dx->getNumCols() + j] \
							- x_CPU[i*2*dE_dx->getNumCols() + j]) / pow(y_CPU[i],2);
				dE_dx_CPU[(i*2+1)*dE_dx->getNumCols()+j] \
					= (x_CPU[i*2*dE_dx->getNumCols() + j] \
							- x_CPU[(i*2+1)*dE_dx->getNumCols() + j]) / pow(y_CPU[i],2);
			}
		}else if(h_labels[i] == 1){
			for(int j=0; j < _fcp->getNumIn(); j++){
				Dtype tmp = 0;
				for(int k=0; k < _fcp->getNumOut(); k++){
					tmp += pow(w_CPU[j*_fcp->getNumOut()+k], 2);
				}
				dE_dx_CPU[i*2*dE_dx->getNumCols()+j] \
					= (x_CPU[(i*2+1)*dE_dx->getNumCols() + j] \
							- x_CPU[i*2*dE_dx->getNumCols() + j])*tmp / pow(y_CPU[i],2);
				dE_dx_CPU[(i*2+1)*dE_dx->getNumCols()+j] \
					= (x_CPU[i*2*dE_dx->getNumCols() + j] \
							- x_CPU[(i*2+1)*dE_dx->getNumCols()+j])*tmp / pow(y_CPU[i],2);
				
			}
		}
	

		if(h_labels[i] == 1){
			
			for(int k=0; k < _fcp->getNumOut(); k++){
				for(int j=0; j < _fcp->getNumIn(); j++){
					dE_dw_CPU[j*_fcp->getNumOut()+k] += -pow(x_CPU[(i*2+1)*dE_dx->getNumCols() + j] \
						- x_CPU[i*2*dE_dx->getNumCols() + j], 2) \
						*w_CPU[j*_fcp->getNumOut()+k] / pow(y_CPU[i], 2);
				}
			}
		}
	}
	for(int k=0; k < _fcp->getNumOut(); k++){
		for(int j=0; j < _fcp->getNumIn(); j++){
			w_CPU[j*_fcp->getNumOut()+k] -= 0.001*dE_dw_CPU[j*_fcp->getNumOut()+k];
		}
	}

	dE_dx->copyFromHost(dE_dx_CPU, dE_dx->getNumEles());


//dE_dx->showValue("Recommendation_dedx");

}


