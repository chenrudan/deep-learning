/*************************************************************************
    > File Name: load_data.h
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com
    > Created Time: 2014年07月17日 星期四 09时02分19秒
 ************************************************************************/

#include <iostream>
#include <vector>

using namespace std;

class Load
{
    public:
        Load();
        ~Load();

/* Function: LoadData
 * ------------------
 * 这个函数返回输入文件的一个包含全部数据的float数组
 */
        vector<float>* LoadData(string filename);
};

