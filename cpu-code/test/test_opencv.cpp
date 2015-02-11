#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "load.h"

using namespace std;
using namespace cv;

int main()
{
//	Load load;
//	vector<unsigned char> *file_data = load.LoadData("../data/stl10/train_X.bin");
//	int k = 0;
	while(true)
	{
	//	Mat image(96, 96, CV_8UC3, Scalar(0,0,255));
/*		for(int i = 0; i < 96; i++)
		{
			for(int j = 0; j < 96; j++)
			{
			    image.at<Vec3b>(i,j)[0] = file_data->at(k);
			    image.at<Vec3b>(i,j)[1] = file_data->at(k + 96*96);
			    image.at<Vec3b>(i,j)[2] = file_data->at(k + 96*96*2);
			//	rgb.val[2] = file_data->at(k);
			//	rgb.val[1] = file_data->at( k + 96*96); 
			//	rgb.val[0] = file_data->at( k + 96*96*2); 
				k++;
			}
		}*/

		Mat img = imread("../data/elephant/image_0001.jpg");
		int w = img.cols;  
		int h = img.rows;
		Mat bk, display;   
		bk.create(h,w,CV_8UC1);  
		bk = Scalar(0);
	//	Mat m1(display,Rect(0,h,w,h));

		vector<Mat> sbgr(img.channels());
		split(img, sbgr);
		vector<Mat> mbgr(img.channels());
	

		mbgr[0] = sbgr[0];  
		mbgr[1] = bk;  
		mbgr[2] = bk;  
//		merge(mbgr);

	/*	cout << img.rows << "\n" << img.cols << endl;
		for(int i=0;i<img.rows;i++)
		{
			for(int j=0;j<img.cols;j++)
			{
				for(int n=0;n<img.channels();n++)
				{
		//			cout << (int)img.at<uchar>(i,j*img.channels()+n) << endl;
				}
			}
		}
		IplImage* img = cvCreateImage(cvSize(96,96), IPL_DEPTH_32F, 3);
		for(int i = 0; i < img->height; i++)
		{
			for(int j = 0; j < img->width; j++)
			{
				((float *)(img->imageData + i*img->widthStep))[j*img->nChannels + 0] = file_data[0][k];
				((float *)(img->imageData + i*img->widthStep))[j*img->nChannels + 1] = file_data[0][k + 96*96];
				((float *)(img->imageData + i*img->widthStep))[j*img->nChannels + 2] = file_data[0][k + 96*96*2];
				k++;
			}
		}		
//	IplImage* img = 0;
//	img = cvLoadImage("../data/elephant/image_0001.jpg");*/
		namedWindow("OutputImage", WINDOW_AUTOSIZE);
		imshow("OutputImage", mbgr[0]);
//		cvShowImage("OutputImage", img);
		waitKey();
	}
	return 0;
}
