/*
 *	filename: load_layer.cpp
 */
#include <cmath>
#include "load_layer.hpp"

using namespace std;

template <typename Dtype>
ImgInfo<Dtype>::ImgInfo(){
	img_train_num = 50000;
    img_test_num = 10000;
    img_channel = 3;
    img_size = 32; 
    img_sqrt = 32*32;  

    train_pixel = new Dtype[img_train_num * img_sqrt * img_channel];
    train_label = new Dtype[img_train_num];
    test_pixel = new Dtype[img_test_num * img_sqrt * img_channel];
    test_label = new Dtype[img_test_num];

    train_pixel_ptr = train_pixel;
    train_label_ptr = train_label;
    test_pixel_ptr = test_pixel;
    test_label_ptr = test_label;
}

template <typename Dtype>
ImgInfo<Dtype>::~ImgInfo(){
	delete[] train_pixel;
	delete[] train_label;
	delete[] test_pixel;
	delete[] test_label;
}



template <typename Dtype>
LoadCifar10<Dtype>::LoadCifar10(ImgInfo<Dtype>* cifar10Info){
	this->_ori_pix = new Dtype[cifar10Info->img_sqrt];	
	this->_img_sqrt = cifar10Info->img_sqrt;
	this->_img_channel = cifar10Info->img_channel; 
}

template <typename Dtype>
LoadCifar10<Dtype>::~LoadCifar10(){
	delete[] this->_ori_pix;
}




template <typename Dtype>
void LoadCifar10<Dtype>::loadBinary(string filename, Dtype* &pixel_ptr, Dtype* &label_ptr){

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
				this->_ori_pix[k] = (int)tmp;
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


template <typename Dtype>
void LoadCifar10<Dtype>::processOneImg(Dtype* pixel_ptr){
	Dtype avg = 0;
	Dtype std = 0;
	for(int i = 0; i < this->_img_sqrt; i++){
		avg += this->_ori_pix[i];
	}
	avg /= this->_img_sqrt;
	
	for(int i = 0; i < this->_img_sqrt; i++){
		pixel_ptr[i] = this->_ori_pix[i] - avg;
		std += pixel_ptr[i] * pixel_ptr[i];
	}
	std = sqrt(std);
	for(int i = 0; i < this->_img_sqrt; i++){
		pixel_ptr[i] /= std;
	}
}











