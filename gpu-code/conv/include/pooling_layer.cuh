/*
 * filename: pooling_layer.cuh
 */
#ifndef POOLING_LAYER_H_
#define POOLING_LAYER_H_

#include <iostream>

#include "utils.cuh"
#include "layer.hpp"
#include "nvmatrix.cuh"

class PoolingLayer : public Layer {

	public:
		PoolingLayer(pars* network);
		~PoolingLayer();

		void initCuda();
		void computeOutputs(NVMatrix* x); 
		void computeDerivsOfInput(NVMatrix* dE_dx);
//		void updatePars();

	private:
		int* _max_pos;

};

#endif

