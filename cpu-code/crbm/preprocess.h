/*************************************************************************
    > File Name: preprocess.h
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com
    > Created Time: 2014年07月14日 星期一 10时41分33秒
 ******************************************************/
#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <iostream>
#include "matrix.h"

using namespace std;

class Preprocess
{
	public:
		Preprocess();
		~Preprocess();

/* Function: BaseWhiten
 * --------------------
 * 白化的静态方法，基本方法，减去均值除以标准差
 * 输入矩阵类
 */
		static void BaseWhiten(Matrix *mat);
};

#endif /*whiten.h*/



