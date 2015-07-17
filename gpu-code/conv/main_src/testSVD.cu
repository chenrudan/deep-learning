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
	Matrix<double>* ori_img = new Matrix<double>(img_size, img_size);


	ori_img->reValue(1.0f);
	
	SVD<double>* svd = new SVD<double>(ori_img, img_size, img_size);

	for (int i = 0; i < img_size; ++i) {
		// u = (cropped_x + [-alpha, 0, ..., 0]') / (cropped_x[0] - alpha)
		svd->computeHouseHolderVecU(i);
		svd->eliminateAForV();
		svd->computeHAndUpdateQ();

		if(i < img_size - 2) {
			svd->computeHouseHolderVecV();
			svd->eliminateAForU();
			svd->computeW();
			svd->computeZ();
			svd->updateA();
			svd->computeGAndUpdateP();
		}
	}
	svd->showB();

	Matrix<double>* B = svd->getPAQ(ori_img);
	B->showValue("paq");

	delete B;

	delete svd;
	return 0;
}


