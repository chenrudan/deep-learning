/*************************************************************************
    > File Name: showthree.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com 
    > Created Time: 2014年09月12日 星期五 19时35分37秒
 ************************************************************************/

#include<iostream>
#include<iostream>
#include<fstream>
#include "/home/crd/crd/deeplearning/opencv/opencv-2.4.9/include/opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

void scaleToUnitInterval(Mat &dispImg, ifstream &fin, int size, int channels, int pos, bool type);

int main()
{
	ifstream fin1("/home/crd/crd/deeplearning/data/whitenface5000.dat", ios::binary);
	ifstream fin2("../data/mnist_layer1_pooling.bin", ios::binary); 
	ifstream fin3("../data/mnist_layer2_in.bin", ios::binary); 
//	ifstream fin1("../data/layer1_0.bin", ios::binary);
//	ifstream fin2("../data/layer1_1.bin", ios::binary);
//	ifstream fin3("../data/layer1_10.bin", ios::binary);
	int num = 40;
	int size = 96;
	int channels = 1;
	for(int i = 0; i < num; i++)
	{
		Mat dispImg;
		
		dispImg.create(Size((size + 1)*channels, 3*(1 + size)), CV_8U);
		
		scaleToUnitInterval(dispImg, fin1, size, channels, 0, false);
		scaleToUnitInterval(dispImg, fin2, size, channels, 1, false);
		scaleToUnitInterval(dispImg, fin3, size, channels, 2, false);

		namedWindow("three", WINDOW_NORMAL);
		imshow("three", dispImg);
		waitKey();
	}
	fin1.close();
	fin2.close();
	fin3.close();

	return 0;
}

void scaleToUnitInterval(Mat &dispImg, ifstream &fin, int size, int channels, int pos, bool type)
{
	for(int j = 0; j < channels; j++)
	{
		Mat tmp(size, size, CV_8U);
		float sum = 0;
		float max_value = -10000;
		float min_value = 10000;
		float *read_value = new float[size*size];
		float sqrt_sum = 0;
		for(int m = 0; m < size; m++)
		{
			for(int n = 0; n < size; n++)
			{
				if(type)
				{
					char buffer;
					fin.read(&buffer, 1);
					if(max_value < buffer)
						max_value = buffer;
					if(min_value > buffer)
						min_value = buffer;
					read_value[m*size + n] = buffer;
//					sum += buffer;
				}
				else
				{
					float buffer;
					fin.read((char *)&buffer, sizeof(float));
					if(max_value < buffer)
						max_value = buffer;
					if(min_value > buffer)
						min_value = buffer;
					read_value[m*size + n] = buffer;
					sum += buffer;
				}
			}
		}
		sum = sum/(size*size); 
		float dist = max_value - min_value;
		for(int m = 0; m < size; m++)
		{
			for(int n = 0; n < size; n++)
			{
				tmp.at<uchar>(n,m) = (read_value[m*size + n] - sum)/dist*255;
				tmp.at<uchar>(n,m) = (read_value[m*size + n] - min_value)/dist*255;
		//		tmp.at<uchar>(n,m) = read_value[m*size + n]*255;
			}
		}
		Mat imgROI = dispImg(Rect(j*(1+size), pos*size + 1, size, size));
    	resize(tmp, imgROI, Size(size, size));
	}
}


























