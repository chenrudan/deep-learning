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
				string filename = "../data/cifar-10-batches-bin/data_batch_"+s+".bin";
				loadBinary(filename, this->_train_pixel_ptr, \
						this->_train_label_ptr);
			}
			loadBinary("../data/cifar-10-batches-bin/test_batch.bin", \
					this->_valid_pixel_ptr, this->_valid_label_ptr);

		}

template <typename Dtype>
void LoadCifar10<Dtype>::loadTrainOneBatch(int batch_idx, int num_process, \
		int pid, Dtype* &mini_pixel, int* &mini_label){
	mini_pixel = this->_train_pixel + batch_idx*_minibatch_size*num_process \
				 *this->_img_channel*this->_img_sqrt \
				 +pid*_minibatch_size*this->_img_channel*this->_img_sqrt;
	mini_label = this->_train_label + batch_idx*_minibatch_size*num_process \
				 + pid*_minibatch_size;
}

//此处的num_process是指除了0进程外的个数
template <typename Dtype>
void LoadCifar10<Dtype>::loadValidOneBatch(int batch_idx, int num_process, \
		int pid, Dtype* &mini_pixel, int* &mini_label){
	mini_pixel = this->_valid_pixel + batch_idx*_minibatch_size*num_process \
				 *this->_img_channel*this->_img_sqrt \
				 +pid*_minibatch_size*this->_img_channel*this->_img_sqrt;
	mini_label = this->_valid_label + batch_idx*_minibatch_size*num_process \
				 + pid*_minibatch_size;
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
			this->stdOneImg(pixel_ptr, this->_img_sqrt);
			if(i != num - 1 || j != this->_img_channel - 1)
				pixel_ptr += this->_img_sqrt;

		}
		if(i != num - 1){
			label_ptr++;
		}
	}
	fin.close();
}

template <typename Dtype>
LoadVOC<Dtype>::LoadVOC(int minibatch_size){

	this->_is_base_alloc = false;

	_train_file = "../data/VOC_train_data.bin";
	_valid_file = "../data/VOC_valid_data.bin";

	ifstream _fin1, _fin2;
	_fin1.open(_train_file.c_str(), ifstream::binary);
	_fin2.open(_valid_file.c_str(), ifstream::binary);
	if(!_fin1.is_open() || !_fin2.is_open()){
		cout << "open original data file failed\n";
		exit(EXIT_FAILURE);
	}
	_fin1.read((char*)&this->_num_train, sizeof(int));
	_fin2.read((char*)&this->_num_valid, sizeof(int));

	_fin1.read((char*)&this->_img_channel, sizeof(int));
	_fin1.read((char*)&this->_img_size, sizeof(int));
	_fin1.read((char*)&this->_img_size, sizeof(int));

	//将valid集偏移到数据的地方统一方法处理
	_fin2.seekg(3*sizeof(int), _fin2.cur);

	this->_img_sqrt = this->_img_size * this->_img_size;

	cout << this->_num_train << ":" << this->_num_valid \
		<< ":" << this->_img_channel \
		<< ":" << this->_img_size << ":" << this->_img_size << endl; 
	_minibatch_size = minibatch_size;

	this->_train_pixel = new Dtype[minibatch_size*this->_img_sqrt*this->_img_channel];
	_object_coord = new int[minibatch_size*4*MAX_OBJECT_NUM];
	this->_train_label = new int[minibatch_size*MAX_OBJECT_NUM];

	///先把label_num全部读出来
	_train_label_num = new int[this->_num_train];
	_valid_label_num = new int[this->_num_valid];
	for(int i = 0; i < this->_num_train; i++){
		_fin1.read((char*)&_train_label_num[i], sizeof(int));
		_fin1.seekg(sizeof(int)*5*_train_label_num[i] \
				+ sizeof(float)*this->_img_channel*this->_img_sqrt, _fin1.cur);
	}
	for(int i = 0; i < this->_num_valid; i++){
		_fin2.read((char*)&_valid_label_num[i], sizeof(int));
		_fin2.seekg(sizeof(int)*5*_valid_label_num[i] \
				+ sizeof(float)*this->_img_channel*this->_img_sqrt, _fin2.cur);
	}
	_fin1.close();
	_fin2.close();
}

template <typename Dtype>
LoadVOC<Dtype>::~LoadVOC(){
	delete[] this->_train_pixel;
	delete[] this->_train_label;
	delete[] _object_coord;
	delete[] _train_label_num;
	delete[] _valid_label_num;
}

template <typename Dtype>
void LoadVOC<Dtype>::loadTrainOneBatch(int batch_idx, \
		int num_process, int pid, Dtype* &mini_pixel, \
		int* &mini_coord){
	loadBinary(_train_file, this->_train_pixel, this->_train_label, \
			_object_coord, \
			_train_label_num, batch_idx, num_process, pid);
	mini_pixel = this->_train_pixel;
	mini_coord = _object_coord;
}

template <typename Dtype>
void LoadVOC<Dtype>::loadValidOneBatch(int batch_idx, \
		int num_process, int pid, Dtype* &mini_pixel, \
		int* &mini_coord){
	loadBinary(_valid_file, this->_train_pixel, this->_train_label, \
			_object_coord, \
			_valid_label_num, batch_idx, num_process, pid);
	mini_pixel = this->_train_pixel;
	mini_coord = _object_coord;
}


//之前的数据集传引用是因为要读全部的数据，所以要留下读取的位置，而本次中
//一次只读取一个minibatch的数据
template <typename Dtype>
void LoadVOC<Dtype>::loadBinary(string filename, Dtype* &pixel_ptr, \
		int* &label_ptr, \
		int* &coord_ptr, int* &label_num, int batch_idx, \
		int num_process, int pid){

	ifstream fin(filename.c_str(), ifstream::binary);		
	memset(coord_ptr, 0, sizeof(int)*_minibatch_size*4*MAX_OBJECT_NUM);
	memset(label_ptr, 0, sizeof(int)*_minibatch_size*MAX_OBJECT_NUM);

	fin.seekg(4*sizeof(int), fin.beg);
	int offset = batch_idx*num_process*_minibatch_size \
				 + pid*_minibatch_size; 
	//本次计算之前所有object个数
	int num_past_object = 0;
	int offidx = 0; 
	while(offidx < offset){
		num_past_object += label_num[offidx];
		offidx++;
	}
	//每记录一个object对应一个label四个coord
	fin.seekg(sizeof(int)*offset + sizeof(int)*5*num_past_object \
			+ offset*this->_img_channel*this->_img_sqrt*sizeof(Dtype), \
			fin.cur);

	for(int i = 0; i < _minibatch_size; i++){

		int num_object;
		fin.read((char*)&num_object, sizeof(int));
		for(int j = 0; j < num_object; j++){
			int tmp;
			//首先是label，再是这个label在原图中的坐标
			fin.read((char*)&tmp, sizeof(int));

			label_ptr[i*MAX_OBJECT_NUM+j] = tmp;
			for(int k = 0; k < 4; k++){
				fin.read((char*)&tmp, sizeof(int));
				_object_coord[i*4*MAX_OBJECT_NUM+j*4+k] = tmp;	
			}
		}
		//然后是像素数据
		for(int j = 0; j < this->_img_channel; j++){
			for(int k = 0; k < this->_img_sqrt; k++){
				fin.read((char*)&pixel_ptr[k], sizeof(Dtype));
			}
			//meanOneImg(pixel_ptr, this->_img_sqrt);
			//	stdOneImg(pixel_ptr, this->_img_sqrt);
			if(i != _minibatch_size - 1 || j != this->_img_channel - 1)
				pixel_ptr += this->_img_sqrt;
		}
	}
	fin.close();
}


template <typename Dtype>
LoadDIC<Dtype>::LoadDIC(int minibatch_size, string train_file, string valid_file){

	this->_is_base_alloc = false;

	_train_file = train_file;
	_valid_file = valid_file;

	ifstream _fin1, _fin2;
	_fin1.open(_train_file.c_str(), ifstream::binary);
	_fin2.open(_valid_file.c_str(), ifstream::binary);

	if(!_fin1.is_open() || !_fin2.is_open()){
		cout << "open original data file failed\n";
		exit(EXIT_FAILURE);
	}

	_fin1.read((char*)&this->_num_train, sizeof(int));
	_fin2.read((char*)&this->_num_valid, sizeof(int));

	_fin1.read((char*)&this->_img_channel, sizeof(int));
	_fin1.read((char*)&this->_img_width, sizeof(int));
	_fin1.read((char*)&this->_img_height, sizeof(int));

	this->_img_sqrt = this->_img_width * this->_img_height;

	cout << this->_num_train << ":" << this->_num_valid \
		<< ":" << this->_img_channel \
		<< ":" << this->_img_height << ":" << this->_img_width << endl; 
	_minibatch_size = minibatch_size;

	this->_train_pixel = new Dtype[minibatch_size*this->_img_sqrt*this->_img_channel];
	this->_train_label = new int[minibatch_size];
	this->_valid_pixel = new Dtype[minibatch_size*this->_img_sqrt*this->_img_channel];
	this->_valid_label = new int[minibatch_size];

	_fin1.close();
	_fin2.close();
}

template <typename Dtype>
LoadDIC<Dtype>::~LoadDIC(){
	delete[] this->_train_pixel;
	delete[] this->_train_label;
	delete[] this->_valid_pixel;
	delete[] this->_valid_label;
}

template <typename Dtype>
void LoadDIC<Dtype>::loadTrainOneBatch(int batch_idx, \
		int num_process, int pid, Dtype* &mini_pixel, \
		int* &mini_label){
	loadBinary(_train_file, this->_train_pixel, this->_train_label, \
			batch_idx, num_process, pid);
	mini_pixel = this->_train_pixel;
	mini_label = this->_train_label;
}

template <typename Dtype>
void LoadDIC<Dtype>::loadValidOneBatch(int batch_idx, \
		int num_process, int pid, Dtype* &mini_pixel, \
		int* &mini_label){
	loadBinary(_valid_file, this->_valid_pixel, this->_valid_label, \
			batch_idx, num_process, pid);
	mini_pixel = this->_valid_pixel;
	mini_label = this->_valid_label;
}


//之前的数据集传引用是因为要读全部的数据，所以要留下读取的位置，而本次中
//一次只读取一个minibatch的数据
template <typename Dtype>
void LoadDIC<Dtype>::loadBinary(string filename, Dtype* pixel_ptr, \
		int* label_ptr, int batch_idx, \
		int num_process, int pid){

	ifstream fin(filename.c_str(), ifstream::binary);		

	fin.seekg(4*sizeof(int), fin.beg);
	int offset = batch_idx*num_process*_minibatch_size \
				 + pid*_minibatch_size; 

	fin.seekg(sizeof(int)*offset \
			+ offset*this->_img_channel*this->_img_sqrt*sizeof(Dtype), \
			fin.cur);

	for(int i = 0; i < _minibatch_size; i++){

		fin.read((char*)&(label_ptr[i]), sizeof(int));

		//然后是像素数据
		for(int j = 0; j < this->_img_channel; j++){
			for(int k = 0; k < this->_img_sqrt; k++){
				fin.read((char*)&pixel_ptr[k], sizeof(Dtype));
			}
			if(i != _minibatch_size - 1 || j != this->_img_channel - 1)
				pixel_ptr += this->_img_sqrt;
		}
	}
	fin.close();
}


template <typename Dtype>
void LoadDICSegment<Dtype>::loadBinary(string filename, Dtype* pixel_ptr, \
		int* label_ptr, int batch_idx, \
		int num_process, int pid){

	ifstream fin(filename.c_str(), ifstream::binary);		

	fin.seekg(4*sizeof(int), fin.beg);
	int offset = batch_idx*num_process*this->_minibatch_size \
				 + pid*this->_minibatch_size; 
	//第一个int是对应原图id，第二个int是对应原图label，第三个是seg之后的本图label
	fin.seekg(sizeof(int)*offset*3 \
			+ offset*this->_img_channel*this->_img_sqrt*sizeof(Dtype), \
			fin.cur);

	for(int i = 0; i < this->_minibatch_size; i++){
		fin.seekg(sizeof(int)*2, fin.cur);

		fin.read((char*)&(label_ptr[i]), sizeof(int));

		//然后是像素数据
		for(int j = 0; j < this->_img_channel; j++){
			for(int k = 0; k < this->_img_sqrt; k++){
				fin.read((char*)&pixel_ptr[k], sizeof(Dtype));
			}
			if(i != this->_minibatch_size - 1 || j != this->_img_channel - 1)
				pixel_ptr += this->_img_sqrt;
		}
	}
	fin.close();
}


template <typename Dtype>
LoadTianchi<Dtype>::LoadTianchi(int minibatch, int num_process, string img_file, \
		string matches_file, string test_file){
	this->_is_base_alloc = false;

	//img_file里面包含了它的id和pixel
	_img_file = img_file;
	_test_file = test_file;
	_matches_file = matches_file;
	_num_process =num_process;

	if(test_file == ""){

		ifstream fin1, fin2;
		fin1.open(_img_file.c_str(), ifstream::binary);
		fin2.open(_matches_file.c_str(), ifstream::binary);

		if(!fin1.is_open() || !fin2.is_open()){
			cout << "open original data file failed\n";
			exit(EXIT_FAILURE);
		}

		int num_matches = 0;
		fin2.read((char*)&num_matches, sizeof(int));

		_num_matches_train = num_matches * 9 / 10;
		_num_matches_valid = num_matches / 10;

		this->_num_train = _num_matches_train*2;
		this->_num_valid = _num_matches_valid*2;

		int num_img = 0;
		fin1.read((char*)&num_img, sizeof(int));
		fin1.read((char*)&this->_img_channel, sizeof(int));
		fin1.read((char*)&this->_img_width, sizeof(int));
		fin1.read((char*)&this->_img_height, sizeof(int));

		this->_img_sqrt = this->_img_width * this->_img_height;
	//	cout << num_matches << ":" << this->_num_train << ":" << this->_num_valid \
			<< ":" << this->_img_channel \
			<< ":" << this->_img_height << ":" << this->_img_width << endl; 
		_minibatch_size = minibatch;
		_matches_batch_size = minibatch / 2 ;

		this->_train_pixel = new Dtype[minibatch*this->_img_sqrt*this->_img_channel*_num_process];
		this->_train_label = new int[_matches_batch_size*_num_process];
		this->_valid_pixel = new Dtype[minibatch*this->_img_sqrt*this->_img_channel*_num_process];
		this->_valid_label = new int[_matches_batch_size*_num_process];

		for(int i=0; i < num_img; i++){
			int tmp = 0;
			fin1.read((char*)&tmp, sizeof(int));
			_img_idx_pos.insert(pair<int, int>(tmp, i));
			fin1.seekg(sizeof(char)*this->_img_sqrt*this->_img_channel, fin1.cur);
		}
		fin1.close();
		fin2.close();

	}else{
		ifstream fin1, fin2;
		fin1.open(_img_file.c_str(), ifstream::binary);
		fin2.open(_test_file.c_str(), ifstream::binary);

		if(!fin1.is_open() || !fin2.is_open()){
			cout << "open original data file failed\n";
			exit(EXIT_FAILURE);
		}
		fin1.read((char*)&_num_train_img, sizeof(int));
		fin2.read((char*)&_num_test_img, sizeof(int));

		this->_num_train = _num_train_img*_num_test_img*2;
//cout << "*****************************" << _num_train_img << ":" << _num_test_img << ":" << this->_num_train << endl;

		fin1.read((char*)&this->_img_channel, sizeof(int));
		fin1.read((char*)&this->_img_width, sizeof(int));
		fin1.read((char*)&this->_img_height, sizeof(int));

		this->_img_sqrt = this->_img_width * this->_img_height;
		_minibatch_size = minibatch;
//		cout << "*******"<< this->_num_train << ":" << this->_img_channel \
			<< ":" << this->_img_height << ":" << this->_img_width << endl; 

		this->_train_pixel = new Dtype[minibatch*this->_img_sqrt*this->_img_channel*_num_process];
		this->_train_label = new int[minibatch*_num_process];

		fin1.close();
		fin2.close();
	}
}

template <typename Dtype>
LoadTianchi<Dtype>::~LoadTianchi(){
	if(_test_file == ""){
		delete[] this->_train_pixel;
		delete[] this->_train_label;
		delete[] this->_valid_pixel;
		delete[] this->_valid_label;
	}
}

template <typename Dtype>
void LoadTianchi<Dtype>::loadTrainOneBatch(int batch_idx, \
		int pid, Dtype* &mini_pixel, int* &mini_label){
	loadBinary("0", this->_train_pixel+pid*(_minibatch_size*this->_img_sqrt*this->_img_channel), \
			this->_train_label+pid*_minibatch_size, batch_idx, pid);
	mini_pixel = this->_train_pixel+pid*(_minibatch_size*this->_img_sqrt*this->_img_channel);
	mini_label = this->_train_label+pid*_minibatch_size;
}

template <typename Dtype>
void LoadTianchi<Dtype>::loadValidOneBatch(int batch_idx, \
		int pid, Dtype* &mini_pixel, \
		int* &mini_label){
	loadBinary("1", this->_valid_pixel+pid*(_minibatch_size*this->_img_sqrt*this->_img_channel), \
			this->_valid_label+pid*_minibatch_size, \
			batch_idx, pid);
	mini_pixel = this->_valid_pixel+pid*(_minibatch_size*this->_img_sqrt*this->_img_channel);
	mini_label = this->_valid_label+pid*_minibatch_size;
}

template <typename Dtype>
void LoadTianchi<Dtype>::loadTestOneBatch(int batch_idx, \
		int pid, Dtype* &mini_pixel, int* &mini_label){
	loadTestNoLabel("2", this->_train_pixel+pid*(_minibatch_size*this->_img_sqrt*this->_img_channel), \
			this->_train_label+pid*_minibatch_size, batch_idx, pid);
	mini_pixel = this->_train_pixel+pid*(_minibatch_size*this->_img_sqrt*this->_img_channel);
	//此处的label是图片的id名称
	mini_label = this->_train_label+pid*_minibatch_size;
	
}

template <typename Dtype>
void LoadTianchi<Dtype>::loadTestNoLabel(string filename, Dtype* pixel_ptr, \
		int *label_ptr, int batch_idx, int pid){
	ifstream fin1(_img_file.c_str(), ifstream::binary);
	ifstream fin2(_test_file.c_str(), ifstream::binary);
		
	if(!fin1.is_open() || !fin2.is_open()){
		cout << "open original data file failed\n";
		exit(EXIT_FAILURE);
	}

	int offset = (batch_idx*_num_process*_minibatch_size/2 \
			+ pid*_minibatch_size/2); 
	for(int i = 0; i < _minibatch_size / 2; i++){
		//分类、第一张图id、第二张图id
		int img1_pos = (offset+i)%_num_train_img;
		int img2_pos = (offset+i)/_num_train_img;
		fin1.seekg(4*sizeof(int)+img1_pos*(sizeof(int) \
					+sizeof(char)*this->_img_channel*this->_img_sqrt), fin1.beg);
		fin1.read((char*)&(label_ptr[2*i]), sizeof(int));
		//然后是像素数据
		unsigned char tmp;
		char buf;
		for(int j = 0; j < this->_img_channel; j++){
			for(int k = 0; k < this->_img_sqrt; k++){
				fin1.read(&buf, 1);
				tmp = buf;
				pixel_ptr[k] = (int)tmp;
			}
			this->meanOneImg(pixel_ptr, this->_img_sqrt);
			this->stdOneImg(pixel_ptr, this->_img_sqrt);
			pixel_ptr += this->_img_sqrt;
		}

		fin2.seekg(4*sizeof(int)+img2_pos*(sizeof(int) \
					+sizeof(char)*this->_img_channel*this->_img_sqrt), fin2.beg);
		fin2.read((char*)&(label_ptr[2*i+1]), sizeof(int));
		for(int j = 0; j < this->_img_channel; j++){
			for(int k = 0; k < this->_img_sqrt; k++){
				fin2.read(&buf, 1);
				tmp = buf;
				pixel_ptr[k] = (int)tmp;
			}
			this->meanOneImg(pixel_ptr, this->_img_sqrt);
			this->stdOneImg(pixel_ptr, this->_img_sqrt);
			if(i != _minibatch_size/2 - 1 || j != this->_img_channel - 1)
				pixel_ptr += this->_img_sqrt;
		}

	}
	fin1.close();
	fin2.close();
}

//之前的数据集传引用是因为要读全部的数据，所以要留下读取的位置，而本次中
//一次只读取一个minibatch的数据
template <typename Dtype>
void LoadTianchi<Dtype>::loadBinary(string filename, Dtype* pixel_ptr, \
		int* label_ptr, int batch_idx, \
		int pid){

	ifstream fin1(_img_file.c_str(), ifstream::binary);
	ifstream fin2(_matches_file.c_str(), ifstream::binary);

	int matches_offset = batch_idx*_num_process*_matches_batch_size \
						 + pid*_matches_batch_size; 
	if(filename == "1")
		matches_offset += _num_matches_train;

	fin2.seekg(sizeof(int)*(4*matches_offset+1), fin2.beg);

	for(int i = 0; i < _matches_batch_size; i++){
		//分类、第一张图id、第二张图id
		int img1_idx, img2_idx;
		fin2.read((char*)&(label_ptr[i]), sizeof(int));
		fin2.seekg(sizeof(float), fin2.cur);
		fin2.read((char*)&img1_idx, sizeof(int));
		fin2.read((char*)&img2_idx, sizeof(int));
		int img1_pos, img2_pos;
		img1_pos = _img_idx_pos.find(img1_idx)->second;
		img2_pos = _img_idx_pos.find(img2_idx)->second;

		fin1.seekg(4*sizeof(int)+img1_pos*(sizeof(int) \
					+sizeof(char)*this->_img_channel*this->_img_sqrt), fin1.beg);
		fin1.seekg(sizeof(int), fin1.cur); 
		//然后是像素数据
		unsigned char tmp;
		char buf;
		for(int j = 0; j < this->_img_channel; j++){
			for(int k = 0; k < this->_img_sqrt; k++){
				fin1.read(&buf, 1);
				tmp = buf;
				pixel_ptr[k] = (int)tmp;
			}
			this->meanOneImg(pixel_ptr, this->_img_sqrt);
			this->stdOneImg(pixel_ptr, this->_img_sqrt);
			pixel_ptr += this->_img_sqrt;
		}

		fin1.seekg(4*sizeof(int)+img2_pos*(sizeof(int) \
					+sizeof(char)*this->_img_channel*this->_img_sqrt), fin1.beg);
		fin1.seekg(sizeof(int), fin1.cur); 
		for(int j = 0; j < this->_img_channel; j++){
			for(int k = 0; k < this->_img_sqrt; k++){
				fin1.read(&buf, 1);
				tmp = buf;
				pixel_ptr[k] = (int)tmp;
			}
			this->meanOneImg(pixel_ptr, this->_img_sqrt);
			this->stdOneImg(pixel_ptr, this->_img_sqrt);
			if(i != _minibatch_size/2 - 1 || j != this->_img_channel - 1)
				pixel_ptr += this->_img_sqrt;
		}

	}
	fin1.close();
	fin2.close();
}











