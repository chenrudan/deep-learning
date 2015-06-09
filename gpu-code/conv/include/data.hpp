/*
 * filename: data.hpp
 */
#ifndef DATA_HPP_
#define DATA_HPP_

#include <vector>

template <typename Dtype>
class Data {

public:
	Data() {}
	virtual ~Data() {}

	void copyFromHost(Dtype* data_value, const int data_len);
	void copyFromDevice(Data* dev_data);
	void copyToHost(Dtype* data_value, const int data_len);
	void copyToDevice(Data* dev_data);

	void zeros();


protected:
	//数据形状不固定，由子类来定
	std::vector<int> _shape;
	Dtype* _data_value;
	Dtype* _diff_value;
	bool _is_own_data;
	bool _is_own_diff;
	int _amount;
};


#endif
