///
/// \file matrix.hpp
/// \brief 继承数据类，拥有矩阵的特性
///

#ifndef Matrix_H_
#define Matrix_H_

#include <iostream>
#include <string>
#include <curand_kernel.h>
#include "cublas_v2.h"
#include "data.hpp"

using namespace std;

/// \brief 实现了矩阵类，数据将以矩阵形式保存
///
template<typename Dtype>
class Matrix : public Data<Dtype> {
private:
    static cudaDeviceProp deviceProps;  ///< 查询gpu硬件规格

public:

    /// 运算的枚举
    ///
    ///	该枚举定义了对类中成员执行何种运算
    enum FUNCTIONS {
        LOG, EXP, RECIPROCAL, SOFTMAX, SIGMOID, DROPOUT
    };

    Matrix(int numRows, int numCols);

    Matrix(const Matrix *like, bool copy);

    Matrix(const Matrix *like);

    ~Matrix();
    /// \brief 初始化类中成员，为行列赋值
	
    void _init(int numRows, int numCols);

    /// \brief 判断两个对象维数是否相等
    inline bool isSameDims(const Matrix<Dtype> *m) const {
        return m->getNumRows() == this->_shape[0] && m->getNumCols() == this->_shape[1];
    }

    inline int getNumRows() const {
        return this->_shape[0];
    }

    inline int getNumCols() const {
        return this->_shape[1];
    }

    inline int getNumEles() const {
        return this->_amount;
    }

    inline void changePtr(const int add) {
        this->_data_value = this->_data_value + add;
    }

    inline void changePtrFromStart(Dtype *start, const int add) {
        this->_data_value = start + add;
    }

    inline void setPtr(Dtype *start) {
        this->_data_value = start;
    }

    /// \brief 求矩阵转置
    void getTranspose(Matrix<Dtype> *target);

    /// \brief 矩阵右乘
    /// \param[in] b
    /// \param[out] target 两个矩阵相乘输出
    void rightMult(Matrix<Dtype> *b, float scale_AB, Matrix<Dtype> *target, \
                cublasHandle_t &handle);

    /// \brief 将每一行累加起来生成一列，列个数保持不变
    /// \param[out] target
    void sumRow(Matrix<Dtype> *target);

    void sumCol(Matrix<Dtype> *target);

    /// \brief 用一个标量减去整个矩阵
    /// \param[out] target 假如没有这个参数，那么计算结果保存在调用矩阵中
    void subtractFromScalar(float scalar, Matrix<Dtype> *target);

    void subtractFromScalar(float scalar);

    /// \brief 矩阵间点乘
    ///
    /// 点乘结果保存在调用矩阵中
    /// \param[in] b 用来与调用矩阵进行点乘
    /// \param[out] target 保存矩阵与列向量点乘，若没有这个参数，则保存在调用矩阵中
    void eltWiseMult(Matrix<Dtype> *b, Matrix<Dtype> *target);

    void eltWiseMult(Matrix<Dtype> *b);

    /// \brief 矩阵每一列与列向量相加
    /// \param[in] vec 用来加法的列向量
    /// \param[out] target 保存矩阵与列向量相加结果，若没有这个参数，则保存在调用矩阵中
    void addColVector(Matrix<Dtype> *vec, float scale_vec, Matrix<Dtype> *target);

    void addColVector(Matrix<Dtype> *vec);

    void addRowVector(Matrix<Dtype> *vec, float scale_vec, Matrix<Dtype> *target);

    void addRowVector(Matrix<Dtype> *vec);

    /// \brief 对矩阵每一个值执行某种运算
    ///
    /// 针对矩阵每一个值，可以执行FUNCTIONS枚举量中任意一种运算
    /// \param[out] target 保存执行运算后的值，没有该参数，则保存在调用矩阵中
    void apply(FUNCTIONS f, Matrix<Dtype> *target);

    void apply(FUNCTIONS f);

	void applyRelu(Matrix<Dtype>* target, int* record, bool direction = true);

	void applyDropout(Matrix<Dtype> *target, Matrix<int>* record, \
		Matrix<curandState>* rand_probs, bool is_set_up);

    /// \brief 矩阵间点加
    ///
    /// 将输入的三个矩阵点加，然后保存在调用矩阵中
    /// \param[in] b 用来与调用矩阵进行点加
    /// \param[in] c 点加
    void addSum(Matrix<Dtype> *b, Matrix<Dtype> *c, float scale_This, \
                float scale_B, float scale_C);

    void add(Matrix<Dtype> *b, float scale_This, float scale);

    /// \brief 矩阵一行最大值
    /// \param[out] max_vec 保存每一行的最大值的位置
    void maxPosInRow(Matrix<Dtype> *max_vec);


    /// \brief 打印矩阵
    /// \param[in] name 矩阵的名称
    void showValue(string name);

    /// \brief 给矩阵重新赋值
    ///
    /// 输入是float时，矩阵全部赋值为这个值。输入是int时，矩阵每个位置对这个int取余
    void reValue(float value);

    void reValue(int value);

    Dtype computeNorm(int len);

    void cropMatToNew(Matrix<Dtype> *tar, const int row_start, const int cropped_height, \
            const int col_start, const int cropped_width);

    Dtype getPosValue(int pos);
    Dtype getFirstPosValue();

    void subedByUnitMat();
    void subPortion(Matrix<Dtype>* b, const int b_row, \
			const int b_col);
    void setValueAt(const int height_idx, \
		const int width_idx, const Dtype value);
};

#include "../src/matrix.cu"

#endif
