/*************************************************************************
    > File Name: cut_pic.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com 
    > Created Time: 2014年08月27日 星期三 10时27分18秒
 ************************************************************************/

#include<iostream>
using namespace std;

#include<iostream>
#include<sstream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "/home/crd/crd/deeplearning/opencv/opencv-2.4.9/include/opencv/cv.h"
#include<dirent.h> 
#include<string> 

using namespace cv;
using namespace std;

vector<string> getFiles(const char *dir);

int main()
{
	vector<string> files;
	string dir = "../../../crd/deeplearning/data/lfw";
	files = getFiles(dir.c_str());
	for(int i = 0; i < files.size(); i++)
	{
		if((files[i] != ".")&&(files[i] != ".."))
		{
			string subdir = "../../../crd/deeplearning/data/lfw/";
			vector<string> subfiles;
			subdir.append(files[i].c_str());
			subfiles = getFiles(subdir.c_str());		
			for(int j = 0; j < subfiles.size(); j++)
			{
				cout << subfiles[j] << endl;			
				if((subfiles[j] != ".")&&(subfiles[j] != ".."))
				{
					string srcAdd = subdir;
					srcAdd.append("/");
					srcAdd.append(subfiles[j]);
					string disAdd = "./faces/";
					stringstream s;
					string str;
					s << i;
					s >> str;
					disAdd.append(str);
					disAdd.append(".jpg");
					const char *pstrImageName = srcAdd.c_str();  
					const char *pstrSaveCutName = disAdd.c_str();        
					double fScale = 0.384;      //缩放倍数  
					CvSize czSize;              //目标图像尺寸  
					  
					//从文件中读取图像    
					IplImage *pSrcImage = cvLoadImage(pstrImageName, CV_LOAD_IMAGE_UNCHANGED);  
					IplImage *pDstImage = NULL;       
					//计算目标图像大小  
					czSize.width = pSrcImage->width*fScale;  
					czSize.height = pSrcImage->height*fScale;        
					//创建图像并缩放  
					pDstImage = cvCreateImage(czSize, pSrcImage->depth, pSrcImage->nChannels);  
					cvResize(pSrcImage, pDstImage, CV_INTER_AREA);   
 								 
					cvSaveImage(pstrSaveCutName, pDstImage); 
		
					cvReleaseImage(&pSrcImage);  
					cvReleaseImage(&pDstImage);  
				}
			}
		}
    }  
    return 0;  
}

vector<string> getFiles(const char *dir)
{
	//打开地址
	vector<string> files;
	DIR *directory_pointer;
	struct dirent *entry;
	directory_pointer=opendir(dir);
	while((entry=readdir(directory_pointer))!=NULL) 
	{ 
		files.push_back(entry->d_name);
	}
	closedir(directory_pointer);
	return files;
}



















