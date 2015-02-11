/*************************************************************************
    > File Name: savefeature.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com 
    > Created Time: 2014年10月17日 星期五 19时27分40秒
 ************************************************************************/

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <sstream>
#include "/home/crd/crd/deeplearning/opencv/opencv-2.4.9/include/opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

void saveL1(int l1_size, int l1_channel, int input_channel, int filter_size, \
			int block_num, int block_size, string filename);

int main()
{
//	int input_size = 28;
	int l1_size = 20;
//	int l2_size = 12;
	int filter_size = 10;
	int input_channel = 1;
	int l1_channel = 3;
//	int l2_channel = 3;
	int block_size = 2;
	int block_num = pow(l1_size, 2)/pow(block_size, 2);
	
	saveL1(l1_size, l1_channel, input_channel, filter_size, block_num, block_size, "../data/mnist_layer1_w.bin");

	
	return 0;
}

void saveL1(int l1_size, int l1_channel, int input_channel, int filter_size, \
			int block_num, int block_size, string filename)
{
	int w_size = pow(l1_size, 2)*l1_channel*pow(filter_size, 2)*input_channel;
	cout << w_size <<endl;
	int w_block_size = pow(block_size, 2)*l1_channel*pow(filter_size, 2)*input_channel;
	float *w = new float[w_size];	
	
	ifstream fin(filename.c_str(), ios::binary);
	fin.read((char *)w, w_size*sizeof(float));
	
	vector<Mat> dis_img;
	for(int i = 0; i < l1_channel; i++)
	{
		Mat tmp;
		tmp.create(Size( l1_size*(filter_size + 1), l1_size*(filter_size + 1)), CV_8U);
		dis_img.push_back(tmp);
	}	
	for(int i = 0; i < block_num; i++)
	{
		for(int channel_idx = 0; channel_idx < l1_channel; channel_idx++)
		{
			for(int block_idx = 0; block_idx < block_size*block_size; block_idx++)
			{
				float max_value = -1000;
				float min_value = 1000;
				int pos = i*w_block_size + channel_idx*w_block_size/l1_channel \
								+ block_idx*pow(filter_size, 2);
								
				Mat tmp(filter_size, filter_size, CV_8U);
								
				for(int m = 0; m < filter_size; m++)
				{
					for(int n = 0; n < filter_size; n++)
					{
						if(max_value < w[pos + m*filter_size + n])
							max_value = w[pos + m*filter_size + n];
						if(min_value > w[pos + m*filter_size + n])
							min_value = w[pos + m*filter_size + n];
					/*	if(w[pos + m*filter_size + n] == 0)
						{
							cout  << i << ":"<<  channel_idx << ":" << block_idx << "\n";
							return;
						}*/
					}
				}
				float dist = max_value - min_value;
				for(int m = 0; m < filter_size; m++)
				{
					for(int n = 0; n < filter_size; n++)
					{
						tmp.at<uchar>(n, m) = (w[pos + m*filter_size + n] - min_value)/dist*255;
					//	cout << (int)tmp.at<uchar>(m, n) << "\t";
					}
				}
				//row表示是哪个process，col表示是哪个thread
				int row = i/(l1_size/block_size);
				int block_row = block_idx/block_size;
				int col = i%(l1_size/block_size);
				int block_col = block_idx%block_size;
				//int real_row = row*(filter_size*block_size + 2) + block_row*(filter_size + 1);
				int real_row=row*block_size+block_row;
				//int real_col = col*(filter_size*block_size + 2) + block_col*(filter_size + 1);
				int real_col=col*block_size+block_col;
				//cout << i << ":"<<  channel_idx << ":" << block_idx << "\n";
				Mat imgROI = dis_img[channel_idx](Rect(real_col*(filter_size+1), real_row*(filter_size+1), filter_size, filter_size));
				resize(tmp, imgROI, Size(filter_size, filter_size));
			}
		}
	}	
	for(int i = 0; i < l1_channel; i++)
	{
		stringstream ss;
		string number;
		ss << i;
		ss >> number;
		string save_name = "./minist1_feature_t1/l1_wh_channel_" + number + ".png";
		imwrite(save_name, dis_img[i]);
	}
}


















