/*
 *	filename: load_layer.hpp
 */
#ifndef LOAD_LAYER_HPP_
#define LOAD_LAYER_HPP_

#include<iostream>
#include<fstream>
#include<vector>
#include<stdlib.h>

using namespace std;

template <typename Dtype>
class ImgInfo{
public:

	ImgInfo();
	~ImgInfo();

	int img_train_num;
	int img_test_num;
	int img_size;
	int img_sqrt;
	int img_channel;
	Dtype* train_label, *train_label_ptr;
	Dtype* test_label, *test_label_ptr;
	Dtype* train_pixel, *train_pixel_ptr;
	Dtype* test_pixel, *test_pixel_ptr;

};

template <typename Dtype>
class LoadLayer {

public:
	
	LoadLayer() {}
	virtual ~LoadLayer() {}
	
		
	/**
	 * Load binary file, pixel, label or one file contain pixel and label.
	 * These call preprocess.
	 */
	virtual Dtype* loadPixel(string filename) {
		return NULL;	
	}
	virtual int* loadLabel(string filename) {
		return NULL;
	}
	virtual void loadBinary(string filename, Dtype* &pixel_ptr, Dtype* &label_ptr) {}
	
	/**
	 * Load picture file
	 */
	virtual void loadPictures() {}
	
	/**
	 * preprocess pixel, input is num * sqrt.
	 */	
	virtual void processOneImg(Dtype* pixel_ptr) {}

protected:
	//length is about one picture
	Dtype* _ori_pix;
	int _img_num;
	int _img_sqrt;
	int _img_channel;
};
	
template <typename Dtype>
class LoadCifar10 : public LoadLayer<Dtype> {

public: 
	LoadCifar10(ImgInfo<Dtype>* cifar10Info);
	~LoadCifar10();

	void loadBinary(string filename, Dtype* &pixel_ptr, Dtype* &label_ptr);	
	void processOneImg(Dtype* pixel_ptr);

};

#include "../src/load_layer.cpp"

#endif









