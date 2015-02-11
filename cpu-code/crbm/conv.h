/*************************************************************************
    > File Name: conv.h
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com
    > Created Time: 2014年07月14日 星期一 16时12分27秒
 ************************************************************************/
#ifndef CONV_H
#define CONV_H

#include <iostream>
#include "matrix.h"

using namespace std;

class Conv
{
    public:
        Conv();
        ~Conv();

/* Function: Conv2d
 * ----------------
 * 返回2d图片输入的卷积，输入包括原始图片、filter图片和间隔
 */
        static Matrix* Conv2d(Matrix *input_image, Matrix *fliter, int step);
};


#endif /*conv.h*/
