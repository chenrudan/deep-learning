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

/// \brief 数据的基本保存单元，一张图片会产生这样的一个对象，但是数据不是它们读取的，它们只保存指针
template <typename Dtype>
class ImgData {

public:
	ImgData() {}
	ImgData(Dtype* pixel, Dtype* label, const int pixel_len) \
		: _pixel(pixel), _label(label), _pixel_len(pixel_len) {}
	~ImgData() {}


	Dtype* getPixel(){
		return _pixel;
	}
	Dtype* getLabel(){
		return _label;
	}

	void swap(const ImgData&);
protected:
	int _pixel_len;
	Dtype* _pixel;
	Dtype* _label;
};



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

	virtual void loadBinary(string filename, Dtype* &pixel_ptr, Dtype* &label_ptr) {}

	void processOneImg(Dtype* pixel_ptr);

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
	Dtype* getTrainLabel(){
		return _train_label;
	}
	Dtype* getValidPixel(){
		return _valid_pixel;
	}
	Dtype* getValidLabel(){
		return _valid_label;
	}
	Dtype* getTestPixel(){
		return _test_pixel;
	}
	Dtype* getTestLabel(){
		return _test_label;
	}

protected:
	int _num_train;
	int _num_valid;
	int _num_test;
	int _img_size;
	int _img_channel;
	int _img_sqrt;

	///返回cpu数据
	Dtype* _train_label;
	Dtype* _valid_label;
	Dtype* _test_label;
	Dtype* _train_pixel;
	Dtype* _valid_pixel;
	Dtype* _test_pixel;
	Dtype* _train_label_ptr;
	Dtype* _valid_label_ptr;
	Dtype* _test_label_ptr;
	Dtype* _train_pixel_ptr;
	Dtype* _valid_pixel_ptr;
	Dtype* _test_pixel_ptr;

	bool _is_base_alloc;

};


template <typename Dtype>
class LoadParticle : public LoadLayer<Dtype> {

public:
	LoadParticle();
	~LoadParticle();

	using LoadLayer<Dtype>::loadBinary;
	void loadBinary(string filename, Dtype* &pixel_ptr, \
			Dtype* &label_ptr, Dtype fixed_label);
	void shuffleComb();

private:

	Dtype* _all_pixel, *_all_pixel_ptr;
	Dtype* _all_label, *_all_label_ptr;
	vector<ImgData<Dtype> > _all_comb;
};


template <typename Dtype>
class LoadCifar10 : public LoadLayer<Dtype> {

public: 
	LoadCifar10(const int num_train, const int num_valid, \
		const int num_test, const int img_size, const int img_channel);

	~LoadCifar10() {}

	void loadBinary(string filename, Dtype* &pixel_ptr, Dtype* &label_ptr);


};

#include "../src/load_layer.cpp"

#endif













