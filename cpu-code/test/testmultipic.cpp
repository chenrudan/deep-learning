/*************************************************************************
    > File Name: testmultipic.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com 
    > Created Time: 2014年08月27日 星期三 16时21分35秒
 ************************************************************************/

#include <iostream>
#include "/home/crd/crd/deeplearning/opencv/opencv-2.4.9/include/opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace std;
using namespace cv;

int main()  
{
	vector<Mat> imgs(4); 
	imgs[0] = imread("../data/faces/2.jpg");
	imgs[1] = imread("../data/faces/3.jpg");
	imgs[2] = imread("../data/faces/4.jpg");
	imgs[3] = imread("../data/faces/5.jpg");  
	//用来最后显示      
    Mat dispImg;  
    
    int row = 4;
    int channels = 1;
  
    int size = 96;  
    int x, y;  
  
    // w - Maximum number of images in a row   
    // h - Maximum number of images in a column    
    // scale - How much we have to resize the image 
 
  	//m,n是图片的位置
    dispImg.create(Size(40 + size*row, 40 + size*channels), CV_8UC3);  
  
    for (int i= 0, m=5, n=5; i<4; i++, m+=(5+size))  
    {  
        x = imgs[i].cols;  
        y = imgs[i].rows;  
 
        if (i%row==0 && m!=5)  
        {  
            m = 5;  
            n += 5+size;  
        }  
        Mat imgROI = dispImg(Rect(m, n, (int)x, (int)y));  
        resize(imgs[i], imgROI, Size((int)x, (int)y));  
    }  
  
    namedWindow("123");  
    imshow("123", dispImg); 
    waitKey();
    return 0; 
}  

