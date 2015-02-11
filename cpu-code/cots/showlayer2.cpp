/*************************************************************************
    > File Name: showlayer2.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com 
    > Created Time: 2014年10月13日 星期一 19时31分49秒
 ************************************************************************/

#include<iostream>
#include<fstream>
#include"opencv/cv.h"
#include"opencv2/core/core.hpp"
#include"opencv2/highgui/highgui.hpp"
extern "C"{
#include "cblas.h"
}

using namespace std;
using namespace cv;

void storeImage2(float* out,int index,int relate_size,int input_channel,int hidden_block,string path,int picture_row,int picture_col);

int main()
{
	const int block_size = 2;
	const int layer1_filter_channel = 3;
	const int layer2_filter_channel = 3;
	const int input_channel = 1;
	const int filter_size = 10;
	const int input_size = 28;
	
	int layer1_out_size = input_size - filter_size + 2;
	int layer2_out_size = layer1_out_size - filter_size + 2;
	int related_input_size = 2*filter_size - 2;
	int w1_block_size = block_size*block_size*layer1_filter_channel*input_channel*filter_size*filter_size;
	int w2_block_size = block_size*block_size*layer2_filter_channel*layer1_filter_channel*filter_size*filter_size;
	int w1_size = layer1_out_size*layer1_out_size*layer1_filter_channel*input_channel*filter_size*filter_size;
	int w2_size = layer2_out_size*layer2_out_size*layer2_filter_channel*input_channel*filter_size*filter_size;
	
	float *layer1_w = new float[w1_size];
	float *layer2_w = new float[w2_size];
	float *recon_w1 = new float[layer1_filter_channel*filter_size*filter_size*related_input_size*related_input_size*input_channel];
	float *stimult = new float[layer2_out_size*layer2_out_size*layer2_filter_channel*related_input_size*related_input_size];
	
	ifstream fin1("../data/mnist_layer1_w.bin", ios::binary);
	ifstream fin2("../data/mnist_layer2_w.bin", ios::binary);
	
	fin1.read((char *)layer1_w, sizeof(float)*w1_size);
	fin2.read((char *)layer2_w, sizeof(float)*w2_size);
		
	
	int layer1_block_num = layer1_out_size/block_size;
	int layer2_block_num = layer2_out_size/block_size;

	int block_related_size = block_size*block_size*related_input_size*related_input_size;
	//m大小6*6
	for(int m = 0; m < layer2_block_num*layer2_block_num; m++)
	{
		int m_row = m/layer2_block_num;
		int m_col = m%layer2_block_num;
		
		//定位到layer2某个2*2*3*10*10*3，用来乘以处理到的18*18
		int layer2_block_w_pos = m_row*layer2_block_num*w2_block_size + m_col*w2_block_size;
/*		//计算layer1的起始block，也就是10*10的起始点
		int layer1_block_start_pos = m_row*layer1_block_num + m_col;		
		int n_row = layer1_block_start_pos/layer1_block_num;
		int n_col = layer1_block_start_pos%layer1_block_num;*/
		for(int i = 0; i < layer1_filter_channel*filter_size*filter_size*related_input_size*related_input_size*input_channel; i++)
		{
				recon_w1[i] = 0.0f; 
		}
		for(int i = m_row; i < m_row + filter_size/block_size; i++)
		{
			for(int j = m_col; j < m_col + filter_size/block_size; j++)
			{
				for(int channel_idx = 0; channel_idx < layer1_filter_channel; channel_idx++)
				{
					for(int point_idx = 0; point_idx < block_size*block_size; point_idx++)
					{
						int origin_pos = i*w1_block_size*layer1_block_num + j*w1_block_size \
							+ channel_idx*block_size*block_size*filter_size*filter_size \
							+ point_idx*filter_size*filter_size;
						//只有10*10*3*18*18做为中间变量
						int new_pos = (i-m_row)*(filter_size/block_size)*block_related_size+ (j-m_col)*block_related_size \
							+ channel_idx*filter_size*filter_size*related_input_size*related_input_size \
							+ point_idx*related_input_size*related_input_size + (i-m_row)*block_size*related_input_size + (j-m_col)*2;
						for(int k = 0; k < filter_size; k++)
						{
							for(int t = 0; t < filter_size; t++)
							{
								if(m == 0)
								{
									cout <<recon_w1[new_pos + k*related_input_size + t] << ":"<< new_pos + k*related_input_size + t<< ":" << m_row <<":" << m_col<< "\n";
									cout << i << ":" << j << ":" << channel_idx << ":" << point_idx << "\n";
								}
								recon_w1[new_pos + k*related_input_size + t] = layer1_w[origin_pos + k*filter_size + t];							
							}
						}
					}
				}
			}
		}
		int stimult_pos = m*layer2_filter_channel*block_size*block_size*related_input_size*related_input_size;
		int M = block_size*block_size*layer2_filter_channel;
		int K = filter_size*filter_size*layer1_filter_channel;
		int N = related_input_size*related_input_size*input_channel;
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, \
					K, 1, layer2_w + layer2_block_w_pos, \
					K, recon_w1, N, 0, stimult + stimult_pos, N);
		storeImage2(stimult + stimult_pos, m, related_input_size, input_channel, block_size*block_size*layer1_filter_channel,"./minist2_feature_t1/", \
					input_channel*block_size*block_size, layer1_filter_channel);
	}		
		
	delete[] layer1_w;
	delete[] layer2_w;
	delete[] stimult;
	delete[] recon_w1;
	return 0;
}

void storeImage2(float* out,int index,int relate_size,int input_channel,int hidden_block,string path,int picture_row,int picture_col){
	int size=relate_size;
	int number=hidden_block;
	for(int i=0;i<number*input_channel;i++){
		float max=-1000;
		float min=1000;
		int zero_count=0;
		for(int j=0;j<size*size;j++){
			if(out[i*size*size+j]>max) max=out[i*size*size+j];
			if(out[i*size*size+j]<min) min=out[i*size*size+j];
			if(out[i*size*size+j]==0) zero_count++;
		}
		//cout<<"i:"<<i<<"\tzero_count:"<<zero_count<<endl;
		float dis=max-min;
		for(int j=0;j<size*size;j++){
			if(dis!=0){ 
				out[i*size*size+j]=(out[i*size*size+j]-min)/dis;
			}
		}
	}
/*	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(0);*/

	Mat dispImg;
	dispImg.create((size+1)*picture_row-1,(size+1)*picture_col-1,CV_8UC1);
	vector<Mat> feature(1);
	for(int i=0;i<picture_col;i++){
		for(int j=0;j<picture_row;j++){
			Mat tmp(size,size,CV_8U);
			for(int m=0;m<size;m++){
				for(int n=0;n<size;n++){
					tmp.at<uchar>(n,m)=out[i*size*size*picture_row+j*size*size+m*size+n]*255;
				}
			}
			Mat imgROI(dispImg,Rect(i*(size+1),j*(size+1),size,size));
			feature[0]=tmp;
			merge(feature,imgROI);
			//Mat roi(display,Rect(m*(w+1),n*(h+1),w,h));
			//resize(tmp,imgROI,Size(size,size));
		}
	}
	stringstream stream1;
	stream1<<index;
	string name=path+"block_"+stream1.str()+".png";
	imwrite(name,dispImg);
	/*
	   string windowname="l2_f_"+stream2.str();
	   namedWindow(windowname,WINDOW_NORMAL);
	   imshow(windowname,dispImg);
	   waitKey();
	   destroyWindow(windowname);*/

}

























