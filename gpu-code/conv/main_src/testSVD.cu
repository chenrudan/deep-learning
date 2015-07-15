/// filename: testSVD.cu
#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cstdio>
#include <string>
#include "matrix.hpp"
#include "svd.hpp"

using namespace std;

int main(){

	//只针对一张图处理
	int img_size = 100;
	int block_size_l = 1;
	Matrix<double>* ori_img = new Matrix<double>(img_size, img_size);


	ori_img->reValue(1.0f);
	
	SVD<double>* svd = new SVD<double>(ori_img, img_size, img_size);

	int k_max = ceil(img_size / block_size_l);
	int vec_u_len, vec_v_len;

	for (int i = 0; i < k_max; ++i) {
		// u = (cropped_x + [-alpha, 0, ..., 0]') / (cropped_x[0] - alpha)
		svd->computeHouseHolderVecU(i);
		vec_u_len = img_size - i;
		vec_v_len = img_size - i - 1;
		// Hi = I - sigma_u * u * u'
		svd->computeH(vec_u_len);
		svd->eliminateAForV(i);

	}

	delete svd;
	/*
	//假设从第一行开始，下标为0，截取出来的矩阵是A(0:m-1,0)，一个列向量
	int crop_idx = 9;
	int crop_col_u_start = crop_idx;
	int crop_u_height = img_size - crop_idx;
	double* cropped_A_for_u = mallocOnGpu(crop_u_height );
	double* householder_vector_u = mallocOnGpu(crop_u_height );

	double scale_one = 1;
	double scale_minus_one = -1;
	double scale_zero = 0;
	kCropImg<<<1, 1024>>>(ori_img, cropped_A_for_u, crop_idx, \
			crop_u_height, crop_col_u_start, 1, img_size);

	computeHouseHolderVecAndAlpha(cropped_A_for_u, crop_u_height , \
	          householder_vector_u, alpha, sigma_u);

	//求v的是个行矢量，使用的时候当成列向量
	int crop_col_v_start = crop_idx + 1;
	int crop_v_width = img_size - crop_col_v_start;
	double* cropped_A_for_v = mallocOnGpu(crop_v_width);
	double* householder_vector_v = mallocOnGpu(crop_v_width);
	kCropImg<<<1, 1024>>>(ori_img, cropped_A_for_v, crop_idx, \
			1, crop_col_v_start, crop_v_width, img_size);
	computeHouseHolderVecAndAlpha(cropped_A_for_v, crop_v_width, \
	          householder_vector_v, beta, sigma_v);

	double *cropped_A_for_z = mallocOnGpu(crop_u_height*crop_v_width);
	kCropImg<<<1, 1024>>>(ori_img, cropped_A_for_z, crop_idx, \
			crop_u_height, crop_col_v_start, crop_v_width, img_size);

	double *x = mallocOnGpu(crop_v_width);

	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 1, crop_v_width, \
	      crop_u_height, &sigma_u, householder_vector_u, 1, \
		  cropped_A_for_z, crop_v_width, &scale_zero, x, 1);

	double* z = mallocOnGpu(crop_v_width);
	double x_mutliply_v = 0;
	cublasDdot(handle, crop_v_width, x, 1, householder_vector_v, 1, &x_mutliply_v);
	double scale_xv = -x_mutliply_v*sigma_v;
	cublasDcopy(handle, crop_v_width, x, 1, z, 1);
	cublasDaxpy(handle, crop_v_width, &scale_xv, householder_vector_v, 1, z, 1);

	double *w = mallocOnGpu(crop_u_height);
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, crop_u_height, \
	      crop_v_width, &sigma_v, householder_vector_v, 1, \
		  cropped_A_for_z, crop_v_width, &scale_zero, w, 1);

	//u*z
	double *delta_A = mallocOnGpu(crop_u_height*crop_v_width);
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, crop_v_width, crop_u_height,
				1, &scale_one, z, crop_v_width, \
		  householder_vector_u, 1, &scale_zero, delta_A, crop_v_width);
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, crop_v_width, crop_u_height,
				1, &scale_one, householder_vector_v, crop_v_width, \
		  w, 1, &scale_one, delta_A, crop_v_width);
	cublasDaxpy(handle, crop_v_width*crop_u_height, &scale_minus_one, \
	          delta_A, 1, cropped_A_for_z, 1);

	double *q = mallocOnGpu(crop_u_height*crop_u_height);
	double *p = mallocOnGpu(crop_v_width*crop_v_width);
	double *delta_q = mallocOnGpu(crop_u_height*crop_u_height);
	double *delta_p = mallocOnGpu(crop_v_width*crop_v_width);

	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, crop_u_height, crop_u_height,
				1, &scale_one, householder_vector_u, crop_u_height, \
		  householder_vector_u, 1, &scale_zero, h, crop_u_height);

	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, crop_v_width, crop_v_width,
				1, &scale_one, householder_vector_v, crop_v_width, \
		  householder_vector_v, 1, &scale_zero, g, crop_v_width);

	double *k = mallocOnGpu(crop_u_height);
	double *l = mallocOnGpu(crop_v_width);

	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, crop_u_height,
				crop_u_height, &sigma_u, householder_vector_u, 1, \
		        q, crop_u_height, &scale_zero, k, 1);


	cout << sigma_v << endl;
//	showValue(z, crop_v_width, 1, "z");
//	showValue(householder_vector_u, crop_u_height, 1, "householder_vector_u");
//	showValue(w, crop_v_width, 1, "w");
	showValue(householder_vector_v, crop_v_width, 1, "householder_vector_v");
//	showValue(delta_A, crop_u_height, crop_v_width, "delta_A");
//	showValue(cropped_A_for_z, crop_u_height, crop_v_width, "cropped_A_for_z");
//	showValue(h, crop_u_height, crop_u_height, "h");
	showValue(g, crop_v_width, crop_v_width, "g");

*/


	return 0;
}


