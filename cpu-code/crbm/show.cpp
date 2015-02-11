/*************************************************************************
    > File Name: show.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com
    > Created Time: 2014年07月18日 星期五 09时22分33秒
 ************************************************************************/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "show.h"
#include "matrix.h"

using namespace std;
using namespace cv;

Show::Show()
{

}

Show::~Show()
{

}

void Show::ShowMyMatrix8U(Matrix* m, int pos)
{
    int row = m->GetRowNum();
    Mat tmp(row, row, CV_8U);
    float min = m->MatrixMin();
    float max = m->MatrixMax();
    Matrix::MatrixAddBias(m, -min);
    m->MatrixMulCoef(1.0/(max+1e-8));
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < row; j++)
        {
            tmp.at<uchar>(i,j) = m->GetElement(i, j)*255.0;
        }
    }
 //   namedWindow("OutputImage", WINDOW_AUTOSIZE);
    namedWindow("OutputImage", WINDOW_NORMAL);
    imshow("OutputImage", tmp);
    stringstream ss;
    ss << pos;
    imwrite("result" + ss.str() +".jpg", tmp);
    waitKey();
    //因为OpenCV中的Mat图像格式文件是BGR的顺序,读取的文件是rgb
}

void Show::ShowMyMatrix32F(Matrix* m)
{
    int row = m->GetRowNum();
    Mat tmp(row, row, CV_32F);
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < row; j++)
        {
            tmp.at<float>(i,j) = m->GetElement(i, j);
        }
    }
   // namedWindow("OutputImage", WINDOW_AUTOSIZE);
    namedWindow("OutputImage", WINDOW_NORMAL);
    imshow("OutputImage", tmp);
    waitKey();
    //因为OpenCV中的Mat图像格式文件是BGR的顺序,读取的文件是rgb
}















