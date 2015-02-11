/*************************************************************************
  > File Name: test.cpp
  > Author: chenrudan
  > Mail: chenrudan123@gmail.com
  > Created Time: 2014年07月13日 星期日 17时09分45秒
 ************************************************************************/

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "matrix.h"
#include "preprocess.h"
#include "conv.h"
#include "crbm.h"
#include "load.h"
#include "show.h"

//#include "/usr/include/python2.7/Python.h"

using namespace std;
using namespace cv;

int main()
{
	Crbm layer_1;
	Crbm layer_2;
	vector<Matrix*> input_image;
	vector<Matrix*> layer1_output_image;
//	vector<Matrix*> layer2_output_image;
	Load load;
	Show show;
	//参数初值
	int batch_size = 2;
	int layer1_filter_size = 15;
	int layer1_image_size = 96;
	int layer1_pooling_size = 2;
	int layer1_input_channels = 3;
	int layer1_channels = 24;

	//int layer2_filter_size = 10;
	//int layer2_image_size = 41;
	//int layer2_pooling_size = 2;
	//int layer2_input_channels = 24;
	//int layer2_channels = 24;

	vector<float> *file_data = load.LoadData("preprocessed.bin");
//	vector<float> *file_data = load.LoadData("show.bin");
//	vector<float> *file_data = load.LoadData("./data_batch_1.bin");
	//给100个图片赋初值
	int k = 0;
	for(int i = 0; i < 50; i++)
	{
		Matrix *p_new_mat = new Matrix[3];
		for(int j = 0; j < 3; j++)
		{
			p_new_mat[j].init(layer1_image_size,layer1_image_size);
			for(int m = 0; m < layer1_image_size; m++)
			{
				for(int n = 0; n < layer1_image_size; n++)
				{
					p_new_mat[j].AddElementByCol(file_data[0][k]);
				//	p_new_mat[j].AddElement(1);
					k++;
				}
			}
   //         float min = (p_new_mat + j)->MatrixMin();
  //          float max = (p_new_mat + j)->MatrixMax();
   //         Matrix::MatrixAddBias((p_new_mat + j), -min);
    //        (p_new_mat + j)->MatrixMulCoef(1.0/(max+1e-8));
		//	show.ShowMyMatrix8U(p_new_mat + j);
		}
		input_image.push_back(p_new_mat);
	}

	int pos = 0;
	/*********************
	*1.初始化参数         *
	*2.训练              *
	**********************/
	int batch_all = 50;
	layer_1.FilterInit(layer1_filter_size, layer1_channels, layer1_input_channels, layer1_image_size, batch_size, layer1_pooling_size);
	cout << "layer1 initialize parameters success!\n";
	int m = 0;
	while(m < 4)
	{
        for(int batch = 0; batch < batch_all/batch_size; batch++)
        {
            cout << "epoch is " << m <<  endl;
            cout << "----------------------------\n";
            vector<Matrix*> *tmp = layer_1.RunBatch(input_image, pos);
            if(batch == batch_all/batch_size - 1)
            {
                layer1_output_image.insert(layer1_output_image.begin(), tmp->begin(), tmp->end());
            }
            pos += 2;
        }
        pos = 0;
        m++;
    }

 /*   pos = 0;
    layer_2.FilterInit(layer2_filter_size, layer2_channels, layer2_input_channels, layer2_image_size, batch_size, layer2_pooling_size);
	cout << "layer2 initialize parameters success!\n";
	for(int batch = 0; batch < batch_all/batch_size; batch++)
	{
	    vector<Matrix*> *tmp = layer_2.RunBatch(layer1_output_image, pos);
	    if(batch == batch_all/batch_size - 1)
	    {
            layer2_output_image.insert(layer2_output_image.end(), tmp->begin(), tmp->end());
	    }
		pos += 2;
	}*/

	/*********************
	*画图显示             *
	**********************/
		//显示权重
	vector<Matrix*> *weight = layer_1.GetWeight();
	int w_size = weight->size();
	for(int i = 0; i < w_size; i++)
	{
	    cout << "weight-----------------------\n";
        show.ShowMyMatrix8U((*weight)[i], i);
	}
/*
	Py_Initialize();
	 // 添加当前路径
    // 把输入的字符串作为Python代码直接运行，返回
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('./')");
    PyObject * pModule = NULL;//声明变量
    PyObject * pFunc = NULL;// 声明变量
    pModule =PyImport_ImportModule("hello");//这里是要调用的文件名
    pFunc= PyObject_GetAttrString(pModule, "tile_raster_images");//这里是要调用的函数名

	//设置方法的参数
    PyObject *pArgs, *pValue *pOut;
    pValue = PyTuple_New(w_size*5*5);
    for(int i = 0; i < w_size; i++)
    {
        for(int m = 0; m < 5; m++)
        {
            for(int n = 0; n < 5; n++)
            {
                PyTuple_SetItem(pValue, i, Py_BuildValue("f", (*weight)[i]->GetElement(m, n)));
            }
        }
    }
    PyTuple_SetItem(pArgs, 0, pValue);
    PyTuple_SetItem(pArgs, 1, Py_BuildValue("(ii)", 5,5));
    PyTuple_SetItem(pArgs, 2, Py_BuildValue("(ii)", 10,10));
    PyTuple_SetItem(pArgs, 3, Py_BuildValue("(ii)", 1,1));
    pOut = PyObject_CallObject(pFunc, pArgs);

    Py_DECREF(pArgs);
    Py_DECREF(pValue);
    Py_DECREF(pOut);
    Py_Finalize();

*/

	//显示第一层输出
/*	ofstream layer1_output("layer1.txt");
	int o_size = layer1_output_image.size();
	cout << o_size << endl;
	for(int i = 0; i < o_size; i++)
	{
	    for(int j = 0; j < layer1_channels; j++)
	    {
            for(int m = 0; m < 14; m++)
            {
                for(int n = 0; n < 14; n++)
                {
                //	cout << output_image[i]->GetElement(m, n) << "\n";
                    layer1_output << layer1_output_image[i][j].GetElement(m, n) << "\n";
                }
            }
	    }
	}
	layer1_output.close();*/

/*	int o_size = layer1_output_image.size();
	for(int i = 0; i < o_size; i++)
	{
	    for(int j = 0; j < layer1_channels; j++)
	    {
	        cout << "outputlayer1-----------------------\n";
	        show.ShowMyMatrix8U(&layer2_output_image[i][j]);
	    }
	}*/
	return 0;
}


