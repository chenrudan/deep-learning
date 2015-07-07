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
LoadParticle<Dtype>::LoadParticle(){

	this->_is_base_alloc = false;

	ifstream fin1("../data/particle/manual-tutorial-positive.bin", \
			ifstream::binary);
	ifstream fin2("../data/particle/manual-tutorial-negative.bin", \
			ifstream::binary);
	int num_pos, num_neg;
	fin1.read((char*)&num_pos, sizeof(int));
	fin2.read((char*)&num_neg, sizeof(int));
	fin1.read((char*)&this->_img_size, sizeof(int));
	
	this->_img_channel = 1;

	cout << num_pos << ":" << num_neg << ":" << this->_img_size \
			<< ":" << this->_img_channel << endl;

	this->_num_train = ceil(((num_neg + num_pos) * 9.0) / 10);
	this->_num_valid = num_neg + num_pos - this->_num_train;
	this->_img_sqrt = this->_img_size * this->_img_size;

	/// 将全部的数据都读进来，然后再处理
	_all_pixel = new Dtype[(num_neg + num_pos) * this->_img_sqrt \
				* this->_img_channel];
	_all_label = new Dtype[num_neg + num_pos];

	_all_pixel_ptr = _all_pixel;
	_all_label_ptr = _all_label;

	fin1.close();
	fin2.close();

	loadBinary("../data/particle/manual-tutorial-positive.bin", \
				_all_pixel_ptr, _all_label_ptr, 1);
	loadBinary("../data/particle/manual-tutorial-negative.bin", \
				_all_pixel_ptr, _all_label_ptr, 0);

	shuffleComb();
	
	this->_train_pixel = _all_comb[0].getPixel();
	this->_train_label = _all_comb[0].getLabel();
	this->_valid_pixel = _all_comb[this->_num_train].getPixel();
	this->_valid_label = _all_comb[this->_num_train].getLabel();

}

template <typename Dtype>
void LoadParticle<Dtype>::shuffleComb(){
	int all_num = this->_num_train + this->_num_valid;
	for(int i = 0; i < all_num; i++){
		int rand_idx1 = rand() % (all_num - 1);
		_all_comb[i].swap(_all_comb[rand_idx1]);	
	}
}


template <typename Dtype>
LoadParticle<Dtype>::~LoadParticle(){
	delete[] _all_pixel;
	delete[] _all_label;
}

template <typename Dtype>
void LoadParticle<Dtype>::loadBinary(string filename, Dtype* &pixel_ptr, \
		Dtype* &label_ptr, Dtype fixed_label){
	ifstream fin(filename.c_str(), ifstream::binary);
	if(!fin.is_open()){
		cout << "open file failed\n";
		exit(EXIT_FAILURE);
	}
	unsigned char tmp;
	char buf;
	int num;
	fin.read((char*)&num, 4);
	fin.seekg(2*sizeof(int), fin.cur);
	
	for(int i = 0; i < num; i++){
		/// 将指针加入容器内
		ImgData<Dtype> my_img = ImgData<Dtype>(pixel_ptr, label_ptr, \
					this->_img_channel * this->_img_sqrt);
		_all_comb.push_back(my_img);
		fin.seekg(2*sizeof(int), fin.cur);

		for(int j = 0; j < this->_img_channel; j++){
			for(int k = 0; k < this->_img_sqrt; k++){
				fin.read(&buf, 1);
				tmp = buf;
				pixel_ptr[k] = (int)tmp;
			}
			processOneImg(pixel_ptr);
			if(i != num - 1 || j != this->_img_channel - 1){
				pixel_ptr += this->_img_sqrt;
			}
		}
		if(i != num - 1){
			*label_ptr = fixed_label;
			label_ptr++;
		}
	}
	fin.close();
}

template <typename Dtype>
void LoadLayer<Dtype>::processOneImg(Dtype* pixel_ptr){
	Dtype avg = 0;
	Dtype std = 0;
	for(int i = 0; i < this->_img_sqrt; i++){

		avg += pixel_ptr[i];
	}
	avg /= this->_img_sqrt;

	for(int i = 0; i < this->_img_sqrt; i++){
		pixel_ptr[i] = pixel_ptr[i] - avg;
		std += pixel_ptr[i] * pixel_ptr[i];
	}
	std /= this->_img_sqrt;
	std = sqrt(std);
	for(int i = 0; i < this->_img_sqrt; i++){
		pixel_ptr[i] /= std;
	}
}

template <typename Dtype>
void ImgData<Dtype>::swap(const ImgData<Dtype>& new_img){
	Dtype* tmp = new Dtype[_pixel_len];
	memcpy(tmp, new_img._pixel, sizeof(Dtype) * _pixel_len);
	memcpy(new_img._pixel, _pixel, sizeof(Dtype) * _pixel_len);
	memcpy(_pixel, tmp, sizeof(Dtype) * _pixel_len);

	memcpy(tmp, new_img._label, sizeof(Dtype));
	memcpy(new_img._label, _label, sizeof(Dtype));
	memcpy(_label, tmp, sizeof(Dtype));

	delete[] tmp;
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
			_train_label = new Dtype[_num_train];
			_train_pixel_ptr = _train_pixel;
			_train_label_ptr = _train_label;
		}
		if (num_valid > 0) {
			_valid_pixel = new Dtype[_num_valid * _img_sqrt * _img_channel];
			_valid_label = new Dtype[_num_valid];
			_valid_pixel_ptr = _valid_pixel;
			_valid_label_ptr = _valid_label;
		}
		if (num_test > 0) {
			_test_pixel = new Dtype[_num_test * _img_sqrt * _img_channel];
			_test_label = new Dtype[_num_test];
			_test_pixel_ptr = _test_pixel;
			_test_label_ptr = _train_label;
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
LoadCifar10<Dtype>::LoadCifar10(const int num_train, const int num_valid, \
		const int num_test, const int img_size, const int img_channel) \
		: LoadLayer<Dtype>(num_train, num_valid, num_test, img_size, img_channel){
		
	for(int i = 1; i < 6; i++){
		string s;
		stringstream ss;
		ss << i;
		ss >> s;
		string filename = "../data/cifar-10-batches-bin/data_batch_"+s+".bin";
		loadBinary(filename, this->_train_pixel_ptr, \
				this->_train_label_ptr);
	}
	loadBinary("../data/cifar-10-batches-bin/test_batch.bin", \
            this->_valid_pixel_ptr, this->_valid_label_ptr);
		
}

template <typename Dtype>
void LoadCifar10<Dtype>::loadBinary(string filename, \
			Dtype* &pixel_ptr, Dtype* &label_ptr){

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
			processOneImg(pixel_ptr);
			if(i != num - 1 || j != this->_img_channel - 1){
				pixel_ptr += this->_img_sqrt;
			}
		}
		if(i != num - 1){
			label_ptr++;
		}
	}
	fin.close();
}



















