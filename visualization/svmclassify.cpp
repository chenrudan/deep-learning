/*************************************************************************
    > File Name: svmclassify.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com 
    > Created Time: 2014年09月23日 星期二 11时03分00秒
 ************************************************************************/

#include<iostream>
#include<iostream>
#include<fstream>
#include "/home/crd/crd/deeplearning/opencv/opencv-2.4.9/include/opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

using namespace std;
using namespace cv;

int main()
{
	// 用于保存可视化数据的矩阵
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);

	// 创建一些训练样本
    ifstream fin("../data/test_out3.dat", ios::binary);
    int num = 3400;
	int postive_num = 1300;
	int size = 42;
	int channels = 8;
	int length = size*size*channels*num;
    float *trainingData = new float[length];
    float *labels = new float[length];
	fin.read((char *)trainingData, length);
	
    for(int i = 0; i < postive_num; i++){
    	labels[i] = 1.0;
    }
    for(int i = postive_num; i < num; i++){
    	labels[i] = -1.0;
    }
	//float labels[4] = {1.0, -1.0, -1.0, -1.0};   

    Mat labelsMat(num, 1, CV_32FC1, labels);
//    float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };

    Mat trainingDataMat(num, 1, CV_32FC1, trainingData);

    // 设置SVM参数
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

    // 对SVM进行训练
    CvSVM SVM;
    SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
    
    Vec3b green(0,255,0), blue (255,0,0);
    // 将SVM断定的分划区域绘制出来
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
        {
            Mat sampleMat = (Mat_<float>(1,2) << i,j);
            float response = SVM.predict(sampleMat);

            if (response == 1)
                image.at<Vec3b>(j, i)  = green;
            else if (response == -1) 
                image.at<Vec3b>(j, i)  = blue;
        }

    // 绘制训练数据点
    int thickness = -1;
    int lineType = 8;
    circle( image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType);
    circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType);
    circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
    circle( image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness, lineType);

    // 绘制支持向量
    thickness = 2;
    lineType  = 8;
    int c     = SVM.get_support_vector_count();

    for (int i = 0; i < c; ++i)
    {
        const float* v = SVM.get_support_vector(i);
        circle( image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), thickness, lineType);
    }

    imwrite("result.png", image);       
    imshow("简单SVM分类", image); 
    waitKey(0);

	return 0;
}

