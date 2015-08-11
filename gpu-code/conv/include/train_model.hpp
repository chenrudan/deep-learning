///
/// \file train_model.hpp
/// \brief 继承数据类，拥有矩阵的特性
///


#ifndef TRAINMODEL_H_
#define TRAINMODEL_H_

#include <iostream>
#include "model_component.hpp"

using namespace std;

/// \brief 实现了网络在训练过程中会执行的一些操作
///
template<typename Dtype>
class TrainModel {
private:


public:
    TrainModel();
    ~TrainModel();

    void parseNetJson(string json_file);

};

#include "../src/matrix.cu"

#endif
