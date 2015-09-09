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

#define MAX_OBJECT_NUM 24

using namespace std;

/// \brief 数据的基本保存单元，一张图片会产生这样的一个对象，但是数据不是它们读取的，它们只保存指针
template <typename Dtype>
class ImgData {

public:
	ImgData() {}
	ImgData(Dtype* pixel, int* label, const int pixel_len) \
		: _pixel(pixel), _label(label), _pixel_len(pixel_len) {}
	~ImgData() {}


	Dtype* getPixel(){
		return _pixel;
	}
	int* getLabel(){
		return _label;
	}

	void swap(const ImgData&);
protected:
	int _pixel_len;
	Dtype* _pixel;
	int* _label;
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

	virtual void loadBinary(string filenmae, Dtype* pixel_ptr, \
		int* label_ptr, int batch_idx, \
		int num_process, int pid) {}

	void meanOneImg(Dtype* pixel_ptr, int process_len);
	void stdOneImg(Dtype* pixel_ptr, int process_len);

	virtual void loadTrainOneBatch(int batch_idx, int num_process, int pid, \
				Dtype* &mini_pixel, int* &mini_label) {}
	virtual void loadValidOneBatch(int batch_idx, int num_process, int pid, \
				 Dtype* &mini_pixel, int* &mini_label) {}

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
	int _num_train;
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
class LoadParticle : public LoadLayer<Dtype> {

public:
	LoadParticle();
	~LoadParticle();

	using LoadLayer<Dtype>::loadBinary;
	void loadBinary(string filename, Dtype* &pixel_ptr, \
			int* &label_ptr, int fixed_label);
	void shuffleComb();

private:

	Dtype* _all_pixel, *_all_pixel_ptr;
	int* _all_label, *_all_label_ptr;
	vector<ImgData<Dtype> > _all_comb;
};


template <typename Dtype>
class LoadCifar10 : public LoadLayer<Dtype> {

	int _minibatch_size;
public: 
	LoadCifar10(const int minibatch_size);

	~LoadCifar10() {}

	using LoadLayer<Dtype>::loadBinary;
	void loadBinary(string filename, Dtype* &pixel_ptr, int* &label_ptr);
	void loadTrainOneBatch(int batch_idx, int num_process, int pid, \
				Dtype* &mini_pixel, int* &mini_label);
	void loadValidOneBatch(int batch_idx, int num_process, int pid, \
				 Dtype* &mini_pixel, int* &mini_label);

};

template <typename Dtype>
class LoadVOC : public LoadLayer<Dtype> {

public: 
	LoadVOC(int minibatch);

	~LoadVOC();

	using LoadLayer<Dtype>::loadBinary;
	void loadBinary(string filenmae, Dtype* &pixel_ptr, \
		int* &label_ptr, \
		int* &coord_ptr, int* &label_num, int batch_idx, \
		int num_process, int pid);
	
	void loadTrainOneBatch(int batch_idx, \
		int num_process, int pid, Dtype* &mini_pixel, \
		int* &mini_coord);
	void loadValidOneBatch(int batch_idx, \
		int num_process, int pid, Dtype* &mini_pixel, \
		int* &mini_coord);

	int* getObjectCoord(){
		return _object_coord;
	}
	int* getTrainLabelNum(){
		return _train_label_num;
	}
	int* getValidLabelNum(){
		return _valid_label_num;
	}

private:
	int _minibatch_size;
	string _train_file;
	string _valid_file;
	int *_object_coord; 
	int *_train_label_num; ///>这个数组保存的minibatch大小数组每一张图中有几个物体label
	int *_valid_label_num;

};

template <typename Dtype>
class LoadDIC : public LoadLayer<Dtype> {

public: 
	LoadDIC(int minibatch, string train_file, string valid_file);

	~LoadDIC();

	void loadBinary(string filenmae, Dtype* pixel_ptr, \
		int* label_ptr, int batch_idx, \
		int num_process, int pid);
	
	void loadTrainOneBatch(int batch_idx, \
		int num_process, int pid, Dtype* &mini_pixel, \
		int* &mini_label);
	void loadValidOneBatch(int batch_idx, \
		int num_process, int pid, Dtype* &mini_pixel, \
		int* &mini_label);

protected:
	int _minibatch_size;
	string _train_file;
	string _valid_file;

};

template <typename Dtype>
class LoadDICSegment : public LoadDIC<Dtype> {

public: 
	LoadDICSegment(int minibatch, string train_file, string valid_file) \
		: LoadDIC<Dtype>(minibatch, train_file, valid_file) {}

	~LoadDICSegment() {}

	void loadBinary(string filenmae, Dtype* pixel_ptr, \
		int* label_ptr, int batch_idx, \
		int num_process, int pid);
private:
	int* _ori_img_label;
	int* _ori_img_idx;
	
};





#include "../src/load_layer.cpp"

#endif













