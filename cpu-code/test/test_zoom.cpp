/*************************************************************************
    > File Name: test_zoom.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com 
    > Created Time: 2014年08月27日 星期三 09时32分10秒
 ************************************************************************/

#include<iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "/home/crd/crd/deeplearning/opencv/opencv-2.4.9/include/opencv/cv.h"

using namespace cv;
using namespace std;

int main()
{
    const char *pstrImageName = "/home/crd/crd/deeplearning/data/lfw/Abel_Aguilar/Abel_Aguilar_0001.jpg";  
    const char *pstrSaveImageName = "001缩放图.jpg";  
    const char *pstrWindowsSrcTitle = "123";  
    const char *pstrWindowsDstTitle = "缩放图 (http://blog.csdn.net/MoreWindows)";       
    double fScale = 0.384;      //缩放倍数  
    CvSize czSize;              //目标图像尺寸  
    		/*for(int i = 1; i <= 435; i++)
		{
			string srcAdd = "../../../crd/deeplearning/randomc101/data/c101/101_ObjectCategories/Faces/";
			string disAdd = "./faces/";
			string tmpAdd = "image_";
			stringstream s;
			string str;
			s << i;
			s >> str;
			if(i/100)
			{
				tmpAdd.append("0");			
			}
			else if(i/10)
			{
				tmpAdd.append("00");
			}
			else
			{
				tmpAdd.append("000");
			}	
			tmpAdd.append(str);			
			tmpAdd.append(".jpg");
			srcAdd.append(tmpAdd);
			disAdd.append(tmpAdd);  */
      
    //从文件中读取图像    
    IplImage *pSrcImage = cvLoadImage(pstrImageName, CV_LOAD_IMAGE_UNCHANGED);  
    IplImage *pDstImage = NULL;         
    //计算目标图像大小  
    czSize.width = pSrcImage->width * fScale;  
    czSize.height = pSrcImage->height * fScale;        
    //创建图像并缩放  
    pDstImage = cvCreateImage(czSize, pSrcImage->depth, pSrcImage->nChannels);  
    cvResize(pSrcImage, pDstImage, CV_INTER_AREA);   
    
    const char *pstrWindowsCutTitle = "456";
    const char *pstrSaveCutName = "cut.jpg";
    IplImage *pCutImage = NULL;
    czSize.width = 96;  
    czSize.height = 96;
    cvSetImageROI(pDstImage,cvRect(0,0,96,96));
    pCutImage = cvCreateImage(cvSize(96,96),  
            pDstImage->depth,  
            pDstImage->nChannels);
    cvCopy(pDstImage,pCutImage,0);
    cvResetImageROI(pDstImage); 
        
    //创建窗口  
    cvNamedWindow(pstrWindowsSrcTitle, CV_WINDOW_AUTOSIZE);  
    cvNamedWindow(pstrWindowsDstTitle, CV_WINDOW_AUTOSIZE); 
    cvNamedWindow(pstrWindowsCutTitle, CV_WINDOW_AUTOSIZE);      
    //在指定窗口中显示图像  
    cvShowImage(pstrWindowsSrcTitle, pSrcImage);  
    cvShowImage(pstrWindowsDstTitle, pDstImage); 
    cvShowImage(pstrWindowsCutTitle, pCutImage);       
    //等待按键事件  
    cvWaitKey();  
      
    //保存图片
    cvSaveImage(pstrSaveImageName, pDstImage); 
    cvSaveImage(pstrSaveCutName, pCutImage); 
      
    cvDestroyWindow(pstrWindowsSrcTitle);  
    cvDestroyWindow(pstrWindowsDstTitle);  
    cvDestroyWindow(pstrWindowsCutTitle);
    cvReleaseImage(&pSrcImage);  
    cvReleaseImage(&pDstImage);  
    cvReleaseImage(&pCutImage);
    
    
    return 0;  
}
