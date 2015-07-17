//
// Created by crd on 15-7-16.
//

#include "matrix.hpp"

#ifndef SVD_HPP
#define SVD_HPP

template <typename Dtype>
class SVD {

public:
    SVD(Matrix<Dtype> *A, const int m, const int n);
    ~SVD();

    void computeHouseHolderVecAndAlpha(const int vec_len, \
          Matrix<Dtype> *householder_vector_gpu, \
          Dtype& alpha_cpu, Dtype& sigma_cpu);

    void computeHouseHolderVecU(const int vec_start_idx);
    void computeHouseHolderVecV();

    void computeH();
    void computeG();

    /// v代表了计算第二维，u代表了计算第一维
    void eliminateAForV();
    void eliminateAForU();

    void computeW();
    void computeZ();


    void updateA();

    void computeHAndUpdateQ();
    void computeGAndUpdateP();

    inline void showB(){
        _A->showValue("B");
    }
    inline void showP(){
        _householder_mat_p->showValue("P");
    }
    inline void showQ(){
        _householder_mat_q->showValue("Q");
    }
    Matrix<Dtype>* getPAQ(Matrix<Dtype> *A);


private:

    /*
     * 以下参数用来计算svd
     */
    /// p: n*n
    /// q: m*m
    Matrix<Dtype>   *_householder_mat_p;
    Matrix<Dtype>   *_householder_mat_q;

    /// 虽然每一次的计算是矢量，但是保存为矩阵形式，有一个L的宽度
    /// u: L*m
    /// v: l*n
    Matrix<Dtype>   *_householder_vec_u;
    Matrix<Dtype>   *_householder_vec_v;

    /// 用来更新A(i+1:m, i+1:n)
    /// A(i+1:m, i+1:n) = A(i+1:m, i+1:n) - u*z' - w*v'
    Matrix<Dtype>   *_w;
    Matrix<Dtype>   *_z;
    Matrix<Dtype>   *_x;

    /// 用来更新A(i, i+1:n)和A(i:m, i)
    Matrix<Dtype>   *_h;
    Matrix<Dtype>   *_g;

    /// alpha是对角化后左上角的元素值
    Dtype _alpha;
    Dtype _sigma_u;
    /// beta对角化后第一行第二个值
    Dtype _beta;
    Dtype _sigma_v;

    /// 在使用cblas库的时候需要的系数
    Dtype _scale_one;
    Dtype _scale_minus_one;
    Dtype _scale_zero;

    int _block_size_l;
    int _height;
    int _width;
    int _vec_u_len;
    int _vec_v_len;
    int _vec_start_idx;

    Matrix<Dtype> *_A;
    Matrix<Dtype> *_cropped_A_for_u_v;
    Matrix<Dtype> *_cropped_A_for_z_w;

    Matrix<Dtype> *_delta_A_for_A;

    cublasHandle_t handle;
};

#include "../src/svd.cu"

#endif //SVD_HPP
