/*************************************************************************
  > File Name: show.cpp
  > Author: chenrudan
  > Mail: chenrudan123@gmail.com 
  > Created Time: 2014年08月27日 星期三 19时05分03秒
 ************************************************************************/

#include<iostream>
#include<fstream>
#include<vector>
#include<sstream>
#include "/home/crd/crd/deeplearning/opencv/opencv-2.4.9/include/opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

void show(string filename, int num, int channels, int size, string windowname);

int main()
{
		show("lfwfaces_test.bin", 400, 1, 96, "./white/whiten");
	//	show("../data/yalefaces/orlfaces.bin", 165, 1, 96, "./white/whiten");
	ifstream fin("lfw_label_test.bin", ios::binary);
//	ifstream fin2("lfwfaces_test.bin", ios::binary);

	char *label = new char[332];
//	float *pix = new float[96*96];
	unsigned char label_tmp;

//	fin.seekg(4, fin.beg);
	fin.read(label, 332);
	for(int i = 0; i < 332; i++){
		label_tmp = label[i];
		cout << (int)label_tmp << endl;
	}
	
//	fin2.seekg(12, fin2.beg);
/*	
	Mat tmp(96, 96, CV_8U);
	for(int m = 0; m < 96; m++)
	{
		for(int n = 0; n < 96; n++)
		{
			double buffer;
			fin2.read((char *)&buffer, sizeof(double));
			tmp.at<uchar>(m,n) = buffer*255;
		}
	}
	imshow("q", tmp);
	waitKey();

	fin.seekg(403, fin.beg);
	fin2.seekg(29417484, fin2.beg);

	fin.read(label, 1);
	label_tmp = label[0];
	cout << (int)label_tmp << endl;
	
//	Mat tmp(96, 96, CV_8U);
	for(int m = 0; m < 96; m++)
	{
		for(int n = 0; n < 96; n++)
		{
			double buffer;
			fin2.read((char *)&buffer, sizeof(double));
			tmp.at<uchar>(m,n) = buffer*255;
		}
	}
	imshow("q", tmp);
	waitKey();
	//	show("../whitenface5000.dat", 100, 1, 96, "./white/whiten");
	//	show("./data/reconstruct.bin", 100, 3, 96, "4epoch");
	//	show("./data/first_w.dat", 100, 8, 20, "4epoch");
*/
	return 0;
}

void show(string filename, int num, int channels, int size, string windowname)
{
	ifstream fin(filename.c_str(), ios::binary);
	for(int i = 0; i < num; i++)
	{
		Mat dispImg;
		dispImg.create(Size((size + 1)*channels, (1 + size)), CV_8U);
		for(int j = 0; j < channels; j++)
		{
			Mat tmp(size, size, CV_8U);
			float sum = 0;
			float max_value = -10000;
			float min_value = 10000;
			float *read_value = new float[size*size];
			for(int m = 0; m < size; m++)
			{
				for(int n = 0; n < size; n++)
				{
					float buffer;
					fin.read((char *)&buffer, sizeof(float));
					//fin1.read(&buffer1, 1);
					if(max_value < buffer)
						max_value = buffer;
					if(min_value > buffer)
						min_value = buffer;
					read_value[m*size + n] = buffer;
					//					cout << buffer << endl;
					//				sum += buffer;
				}
			}
			//		sum = sum/(size*size); 
			float dist = max_value - min_value;
			for(int m = 0; m < size; m++)
			{
				for(int n = 0; n < size; n++)
				{
					tmp.at<uchar>(m,n) = (read_value[m*size + n] - min_value)/dist*255;
				}
			}
			Mat imgROI = dispImg(Rect(j*(1+size), 0, size, size));  
			resize(tmp, imgROI, Size(size, size));
		}
		vector<int> compression_params;
		compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
		compression_params.push_back(0);

		//namedWindow(windowname.c_str(), WINDOW_NORMAL);
		stringstream stream1;
		stream1<<i;
		string name= windowname + stream1.str()+".png";             
		imwrite(name,dispImg);

		//waitKey();
	}
	fin.close();
}











