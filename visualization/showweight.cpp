/*************************************************************************
    > File Name: showweight.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com 
    > Created Time: 2014年09月24日 星期三 20时46分58秒
 ************************************************************************/

#include<iostream>
#include<iostream>
#include<fstream>
#include<sstream>
#include "/home/crd/crd/deeplearning/opencv/opencv-2.4.9/include/opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

void scaleToUnitInterval(Mat &dispImg, ifstream &fin, int size, int channels, int block_num);

int main()
{
	ifstream fin1("../data/mnist_layer1_1.bin", ios::binary);
	
	int num = 10;
	int block_num = 2*2;
	int size = 10;
	int channels = 3;
	for(int i = 0; i < num; i++)
	{
		Mat dispImg;
		
		dispImg.create(Size((size + 1)*channels, block_num*(1 + size)), CV_8U);
		
		scaleToUnitInterval(dispImg, fin1, size, channels, block_num);
		
		vector<int> compression_params;
		compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
		compression_params.push_back(0);
		
		stringstream stream1;
		stream1 << i;
		string name = "./mnist1_w_1/block_"+stream1.str()+".png";		
		imwrite(name,dispImg);		
	}
	fin1.close();

	return 0;
}

void scaleToUnitInterval(Mat &dispImg, ifstream &fin, int size, int channels, int block_num)
{
	for(int j = 0; j < channels; j++)
	{
		for(int i = 0; i < block_num; i++)
		{
			Mat tmp(size, size, CV_8U);			
			float max_value = -100;
			float min_value = 100;
			float *buffer = new float[size*size];
			
			fin.read((char *)buffer, sizeof(float)*size*size);
		
			for(int m = 0; m < size; m++)
			{
				for(int n = 0; n < size; n++)
				{		
					buffer[m*size + n] = buffer[m*size + n];				
					if(max_value < buffer[m*size + n])
						max_value = buffer[m*size + n];
					if(min_value > buffer[m*size + n])
						min_value = buffer[m*size + n];		
				}
			}
			float dist = max_value - min_value;
			for(int m = 0; m < size; m++)
			{
				for(int n = 0; n < size; n++)
				{
					tmp.at<uchar>(n,m) = (buffer[m*size + n] - min_value)/dist*255;
				}
			}
			Mat imgROI = dispImg(Rect(j*(1 + size), i*(size + 1), size, size));
			resize(tmp, imgROI, Size(size, size));
			delete[] buffer;
    	}	
	}
}


