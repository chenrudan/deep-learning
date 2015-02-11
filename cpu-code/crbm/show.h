/*************************************************************************
    > File Name: show.h
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com
    > Created Time: 2014年07月18日 星期五 09时22分27秒
 ************************************************************************/

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "matrix.h"

using namespace std;

class Show
{
    public:
        Show();
        ~Show();

/* Function: ShowIamgeWithMat
 * ---------------------------
 * 这个函数返回输入mat形成的图片
 */
        void ShowMyMatrix8U(Matrix* m, int pos);
        void ShowMyMatrix32F(Matrix* m);
};

