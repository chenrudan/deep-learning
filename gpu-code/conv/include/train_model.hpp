///
/// \file train_model.hpp
/// \brief 继承数据类，拥有矩阵的特性
///


#ifndef TRAINMODEL_H_
#define TRAINMODEL_H_

#include <iostream>
#include <string>


using namespace std;

/// \brief 实现了网络在训练过程中会执行的一些操作
///
template<typename Dtype>
class Matrix : public Data<Dtype> {
private:


public:


};

#include "../src/matrix.cu"

#endif
