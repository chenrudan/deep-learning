/*
 * filename: pooling_layer.cu
 */

#include "pooling_layer.cuh"
#include "layer_kernel.cuh"
#include <cmath>

using namespace std;

PoolingLayer::PoolingLayer(pars* network){
	this->_in_size                  = network->in_size;
	this->_in_channel               = network->in_channel;
	this->_pool_size				= network->pool_size;
	this->_stride					= network->stride;
	this->_out_size					= ceil(((_in_size - _pool_size)*1.0f) / _stride) + 1;

	//w_hk的learning rate
	this->_w_lr                     = network->w_lr;
	//out bias learning rate
	this->_b_lr                     = network->b_lr;
	//上一次更新的参数控制增长趋势
	this->_momentum                 = network->momentum;
	this->_weight_decay             = network->weight_decay;

	this->_minibatch_size           = network->minibatch_size;
	this->_lr_down_scale            = network->lr_down_scale;

	cublasCreate(&handle);
}

PoolingLayer::~PoolingLayer() {

	//	delete _w;
	//	delete _w_inc;
	//	delete _bias;
	//	delete _bias_inc;

	delete  _y;
	delete  _dE_dy;
	//	delete _dE_db;
	//	delete _dE_dw;
	cudaFree(_max_pos);
	cublasDestroy(handle);
}

void PoolingLayer::initCuda() {

	//	this->_w            = new NVMatrix(_num_in, _num_out);
	//                  NVMatrix::ALLOC_ON_UNIFIED_MEMORY);
	//	this->_bias         = new NVMatrix(1, _num_out);
	//                  NVMatrix::ALLOC_ON_UNIFIED_MEMORY);

	this->_y               = new NVMatrix(_minibatch_size, \
					_out_size * _out_size * _in_channel);

	this->_dE_dy           = new NVMatrix(_y);
	//	this->_dE_db           = new NVMatrix(_bias);
	//	this->_dE_dw          = new NVMatrix(_w);

	//	this->_w_inc         = new NVMatrix(_w);
	//	this->_bias_inc        = new NVMatrix(1, _num_out);
	//	this->_w_inc->zeros();
	//	this->_bias_inc->zeros();

	cudaError_t status = cudaMalloc((void**) &_max_pos, \
			_minibatch_size * _in_channel * _out_size * _out_size * sizeof(int));
	if (status != cudaSuccess) {
		fprintf(stderr, "!!!! device memory allocation error\n");
		exit(EXIT_FAILURE);
	}

}

void PoolingLayer::computeOutputs(NVMatrix* x){
	dim3 blocks = dim3(_minibatch_size, _in_channel);
	dim3 threads = dim3(ceil(_out_size / 16.0) * 16,  ceil(_out_size / 16.0) * 16);
	//24*24,pooling到12*12
/*	x->reValue(27);
	int length = _minibatch_size * _in_size * _in_size * _in_channel;
	float* tmp = new float[length];
	memset(tmp, 0, sizeof(float) * length);
	tmp[4] = 1;
	cudaMemcpy(x->getDevData(), tmp, sizeof(float) * length, cudaMemcpyHostToDevice);
*/
	max_pooling<<<blocks, threads, sizeof(float)*_pool_size*_out_size*_pool_size*_out_size>>>(x->getDevData(), \
			_y->getDevData(), _max_pos, _in_size, _out_size, _pool_size, _stride);  
	cudaThreadSynchronize();

//x->showValue("x");
//_y->showValue("y");


}

void PoolingLayer::computeDerivsOfInput(NVMatrix* dE_dx){

	dim3 blocks = dim3(_minibatch_size, _in_channel);
	dim3 threads = dim3(ceil(_out_size / 16.0) * 16,  ceil(_out_size / 16.0) * 16);
	//dE_dy_h, 16*16*24*24
	dE_dx->zeros();
//_dE_dy->reValue(13);
	
/*
int length = dE_dx->getNumRows()*dE_dx->getNumCols();
int* tmp = new int[length];
cudaMemcpy(tmp, _max_pos, sizeof(int)*length, cudaMemcpyDeviceToHost);
cout << "maxpos\n";
for(int i = 0; i < length; i++){
	cout << tmp[i] << " ";
}*/

	compute_dE_dy_max<<<blocks, threads, sizeof(float)*_in_size*_in_size>>>(_dE_dy->getDevData(), \
			dE_dx->getDevData(), _max_pos, _in_size, \
			_out_size, _pool_size, _stride);
	cudaThreadSynchronize();

}

/*
   void ConvNet::computeAvgOutputs(){
	   //16*16
	   dim3 blocks = dim3(_minibatch_size, _in_channels);
	   dim3 threads = dim3(_out_size, _out_size);
	   //24*24,pooling到12*12
	   avg_pooling<<<blocks, threads>>>(_y_h->getDevData(), _y_i->getDevData());
	   cudaThreadSynchronize();
   }*/




