///
/// \file load_layer.hpp
/// \brief 从文件中下载数据
///

#ifndef LOAD_LAYER_HPP_
#define LOAD_LAYER_HPP_

#include<iostream>
#include<fstream>
#include<vector>
#include<map>
#include<stdlib.h>
#include"utils.cuh"

#define MAX_OBJECT_NUM 24

using namespace std;

/// \brief 执行下载数据行为的类
///
template <typename Dtype>
class LoadLayer {

public:

	/// \brief 默认构造函数表示个数信息需要从文件中读取，而不是传递进来的
	LoadLayer() {}
	LoadLayer(const int num_train, const int num_valid, \
		const int num_test, const int img_size, const int img_channel);
	virtual ~LoadLayer();

	virtual void loadBinary(string filenmae, Dtype* pixel_ptr, \
		int* label_ptr, int batch_idx) {}

	void meanOneImg(Dtype* pixel_ptr, int process_len);
	void stdOneImg(Dtype* pixel_ptr, int process_len);

	virtual void loadTrainOneBatch(int batch_idx, \
				Dtype* &mini_pixel, int* &mini_label) {}
	virtual void loadValidOneBatch(int batch_idx, \
				 Dtype* &mini_pixel, int* &mini_label) {}
	virtual void loadTestOneBatch(int batch_idx, \
				Dtype* &mini_pixel, int *&mini_label) {}

	int getNumTrain(){
		return _num_train;
	}
	int getNumValid(){
		return _num_valid;
	}
	int getNumTest(){
		return _num_test;
	}
	int getImgSize(){
		return _img_size;
	}
	int getImgChannel(){
		return _img_channel;
	}

	Dtype* getTrainPixel(){
		return _train_pixel;
	}
	int* getTrainLabel(){
		return _train_label;
	}
	Dtype* getValidPixel(){
		return _valid_pixel;
	}
	int* getValidLabel(){
		return _valid_label;
	}
	Dtype* getTestPixel(){
		return _test_pixel;
	}
	int* getTestLabel(){
		return _test_label;
	}

protected:
	long long _num_train;
	int _num_valid;
	int _num_test;
	int _img_size;
	int _img_height;
	int _img_width;
	int _img_channel;
	int _img_sqrt;

	///返回cpu数据
	int* _train_label;
	int* _valid_label;
	int* _test_label;
	Dtype* _train_pixel;
	Dtype* _valid_pixel;
	Dtype* _test_pixel;
	int* _train_label_ptr;
	int* _valid_label_ptr;
	int* _test_label_ptr;
	Dtype* _train_pixel_ptr;
	Dtype* _valid_pixel_ptr;
	Dtype* _test_pixel_ptr;

	bool _is_base_alloc;

};


template <typename Dtype>
class LoadCifar10 : public LoadLayer<Dtype> {

	int _minibatch_size;
public: 
	LoadCifar10(const int minibatch_size);

	~LoadCifar10() {}

	using LoadLayer<Dtype>::loadBinary;
	void loadBinary(string filename, Dtype* &pixel_ptr, int* &label_ptr);
	void loadTrainOneBatch(int batch_idx, 
				Dtype* &mini_pixel, int* &mini_label);
	void loadValidOneBatch(int batch_idx, 
				 Dtype* &mini_pixel, int* &mini_label);

};


#include "../src/load_layer.cpp"

#endif













