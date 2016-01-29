/*
 *	filename: load_layer.cpp
 */
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <bits/stl_bvector.h>
#include <algorithm>
#include "load_layer.hpp"

using namespace std;

template <typename Dtype>
void LoadLayer<Dtype>::meanOneImg(Dtype* pixel_ptr, int process_len){
	Dtype avg = 0;
	for(int i = 0; i < process_len; i++){
		avg += pixel_ptr[i];
	}
	avg /= process_len;

	for(int i = 0; i < process_len; i++){
		pixel_ptr[i] = pixel_ptr[i] - avg;
	}
}

template <typename Dtype>
void LoadLayer<Dtype>::stdOneImg(Dtype* pixel_ptr, int process_len){
	Dtype std = 0;
	for(int i = 0; i < process_len; i++){
		std += pixel_ptr[i] * pixel_ptr[i];
	}

	std /= process_len;
	std = sqrt(std);
	for(int i = 0; i < process_len; i++){
		pixel_ptr[i] /= std;
	}
}

template <typename Dtype>
LoadLayer<Dtype>::LoadLayer(const int num_train, const int num_valid, \
		const int num_test, const int img_size, const int img_channel) \
	: _num_train(num_train), _num_test(num_test), _num_valid(num_valid), \
	_img_size(img_size), _img_channel(img_channel){
		_img_sqrt = _img_size * _img_size;
		if (img_size > 0 && img_channel > 0) {
			if (num_train > 0) {
				_train_pixel = new Dtype[_num_train * _img_sqrt * _img_channel];
				_train_label = new int[_num_train];
				_train_pixel_ptr = _train_pixel;
				_train_label_ptr = _train_label;
			}
			if (num_valid > 0) {
				_valid_pixel = new Dtype[_num_valid * _img_sqrt * _img_channel];
				_valid_label = new int[_num_valid];
				_valid_pixel_ptr = _valid_pixel;
				_valid_label_ptr = _valid_label;
			}
			if (num_test > 0) {
				_test_pixel = new Dtype[_num_test * _img_sqrt * _img_channel];
				_test_label = new int[_num_test];
				_test_pixel_ptr = _test_pixel;
				_test_label_ptr = _test_label;
			}
		}
		_is_base_alloc = true;

	}

template <typename Dtype>
LoadLayer<Dtype>::~LoadLayer(){
	if (_img_size > 0 && _img_channel > 0 && _is_base_alloc == true) {
		if (_num_train > 0) {
			delete[] _train_pixel;
			delete[] _train_label;
		}
		if (_num_valid > 0) {
			delete[] _valid_pixel;
			delete[] _valid_label;
		}
		if (_num_test > 0) {
			delete[] _test_pixel;
			delete[] _test_label;
		}
	}
}

template <typename Dtype>
LoadCifar10<Dtype>::LoadCifar10(const int minibatch_size) : \
		LoadLayer<Dtype>(50000, 10000, 0, 32, 3){

			_minibatch_size = minibatch_size;

			for(int i = 1; i < 6; i++){
				string s;
				stringstream ss;
				ss << i;
				ss >> s;
				string filename = "../../data/cifar-10-batches-bin/data_batch_"+s+".bin";
				loadBinary(filename, this->_train_pixel_ptr, \
						this->_train_label_ptr);
			}
			loadBinary("../../data/cifar-10-batches-bin/test_batch.bin", \
					this->_valid_pixel_ptr, this->_valid_label_ptr);

		}

template <typename Dtype>
void LoadCifar10<Dtype>::loadTrainOneBatch(int batch_idx, \
		Dtype* &mini_pixel, int* &mini_label){
	mini_pixel = this->_train_pixel + batch_idx*_minibatch_size \
				 *this->_img_channel*this->_img_sqrt;
	mini_label = this->_train_label + batch_idx*_minibatch_size;
}


template <typename Dtype>
void LoadCifar10<Dtype>::loadValidOneBatch(int batch_idx, \
		Dtype* &mini_pixel, int* &mini_label){
	mini_pixel = this->_valid_pixel + batch_idx*_minibatch_size \
				 *this->_img_channel*this->_img_sqrt;
	mini_label = this->_valid_label + batch_idx*_minibatch_size;
}

template <typename Dtype>
void LoadCifar10<Dtype>::loadBinary(string filename, \
		Dtype* &pixel_ptr, int* &label_ptr){

	ifstream fin(filename.c_str(), ifstream::binary);		
	if(!fin.is_open()){
		cout << "open file failed\n";
		exit(EXIT_FAILURE);
	}
	unsigned char tmp;
	char buf;
	fin.seekg(0, fin.end);
	int length = fin.tellg();
	int num = length / (this->_img_sqrt * this->_img_channel + 1);
	//numebr of picture in this input file. 
	fin.seekg(0, fin.beg);

	for(int i = 0; i < num; i++){
		fin.read(&buf, 1);
		tmp = buf;
		label_ptr[0] = (int)tmp;
		for(int j = 0; j < this->_img_channel; j++){
			for(int k = 0; k < this->_img_sqrt; k++){
				fin.read(&buf, 1);
				tmp = buf;
				pixel_ptr[k] = (int)tmp;
			}
			this->meanOneImg(pixel_ptr, this->_img_sqrt);
//			this->stdOneImg(pixel_ptr, this->_img_sqrt);
			if(i != num - 1 || j != this->_img_channel - 1)
				pixel_ptr += this->_img_sqrt;

		}
		if(i != num - 1){
			label_ptr++;
		}
	}
	fin.close();
}









