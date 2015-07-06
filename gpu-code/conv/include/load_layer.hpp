///
/// \file load_layer.hpp
/// \brief 从文件中下载数据
///

#ifndef LOAD_LAYER_HPP_
#define LOAD_LAYER_HPP_

#include<iostream>
#include<fstream>
#include<vector>
#include<stdlib.h>

using namespace std;

/// \brief 执行下载数据行为的类
///
template <typename Dtype>
class LoadLayer {

public:
	
	LoadLayer();
	virtual ~LoadLayer();

	virtual void loadBinary(string filename, Dtype* &pixel_ptr, Dtype* &label_ptr) {}

	void processOneImg(Dtype* pixel_ptr);

};

/// \brief 保存了需要进行运算的所有数据集的信息，但是不进行操作
template <typename Dtype>
class DatasetInfo {

public:
	/// \brief 默认构造函数表示个数信息需要从文件中读取，而不是传递进来的
	DatasetInfo();
	DatasetInfo(const int num_train, const int num_valid, \
		const int num_test, const int img_size, const int img_channel);
	~DatasetInfo();

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

protected:

	int _num_train;
	int _num_valid;
	int _num_test;
	int _img_size;
	int _img_channel;
	int _img_sqrt;

	///返回cpu数据
	Dtype* _train_label, *_train_label_ptr;
	Dtype* _valid_label, *_valid_label_ptr;
	Dtype* _test_label, *_test_label_ptr;
	Dtype* _train_pixel, *_train_pixel_ptr;
	Dtype* _valid_pixel, *_valid_pixel_ptr;
	Dtype* _test_pixel, *_test_pixel_ptr;

	friend class LoadLayer;

};

/// \brief 数据的基本保存单元，一张图片会产生这样的一个对象，但是数据不是它们读取的，它们只保存指针
template <typename Dtype>
class ImgData {

public:
	ImgData(const Dtype* pixel, const Dtype* label) \
		: _pixel(pixel), _label(label) {}
	~ImgData() {}

protected:
	Dtype* _pixel;
	Dtype* _label;
};


template <typename Dtype>
class LoadParticle : public LoadLayer<Dtype> {

public:
	LoadParticle();
	~LoadParticle() {}

	using LoadLayer<Dtype>::loadBinary;
	void loadBinary(string filename, Dtype* &pixel_ptr);

private:
	DatasetInfo* _particle_DatasetInfo;

	Dtype* _all_pixel;
	Dtype* _all_label;
	vector<ImgData> _all_comb;
};



/*template <typename Dtype>
class ImgDataIterator {

private:
	vector<Dtype*>

public:
	typedef random_access_iterator_tag iterator_category;
	typedef vector<Dtype*> value_type;
	typedef prtdiff_t difference_type;
	typedef vector<Dtype*>* pointer;
	typedef vector<Dtype*>& reference;
};*/


template <typename Dtype>
class LoadCifar10 : public LoadLayer<Dtype> {

public: 
	LoadCifar10();
	~LoadCifar10();

	/// @brief 一次性读取文件中数据，成功读取返回true，失败返回false
	void loadOnce();
	void loadBinary(string filename, Dtype* &pixel_ptr, Dtype* &label_ptr);

private:
	DatasetInfo* _cifar10_DatasetInfo;
};




#include "../src/load_layer.cpp"

#endif













