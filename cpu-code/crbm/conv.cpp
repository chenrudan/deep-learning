/*************************************************************************
    > File Name: conv.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com
    > Created Time: 2014年07月14日 星期一 16时13分56秒
 ************************************************************************/

#include<iostream>
#include"conv.h"
#include "matrix.h"

using namespace std;

Conv::Conv()
{
}

Conv::~Conv()
{
}

Matrix* Conv::Conv2d(Matrix *input_image, Matrix *fliter, int step)
{
    int prod_row = (input_image->GetRowNum() - fliter->GetRowNum())/step + 1;
    Matrix *prod_mat = new Matrix(prod_row, prod_row);
    //按照得到的feature map来进行顺序计算
    for(int i = 0; i < prod_row*prod_row; i++)
    {
        float element = 0;
        for(int row = 0; row < fliter->GetRowNum(); row++)
        {
            for(int col = 0; col < fliter->GetRowNum(); col++)
            {
                element += input_image->GetElement((i/prod_row)*step + row, col + (i%prod_row)*step) \
                      *fliter->GetElement(row, col);
            }
        }
        prod_mat->AddElement(element);
    }
    return prod_mat;
}

















