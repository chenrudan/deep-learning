#include<iostream>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

#define filterWidth 3 
#define filterHeight 3 
       
double filter[filterWidth][filterHeight] =  
{ 
     0, 0, 0, 
     0, 1, 0, 
     0, 0, 0 
}; 
       
double factor = 1.0; 
double bias = 0.0; 
       
int main(int argc, char *argv[]) 
{ 
	Mat ori_img = imread("test.jpg");
	cout << ori_img.type() << endl;
	Mat out_img;
	out_img.create(ori_img.size(), CV_32SC3);
	cout << out_img.type() << endl;
	imshow("ori", ori_img);

	for(int x=0; x<ori_img.cols; x++){
		for(int y=0; y<ori_img.rows; y++){
        	double red = 0.0, green = 0.0, blue = 0.0; 
	
        	for(int filterX = 0; filterX < filterWidth; filterX++){
		        for(int filterY = 0; filterY < filterHeight; filterY++){
            		int imageX = (x - filterWidth / 2 + filterX + ori_img.cols) % ori_img.cols; 
		            int imageY = (y - filterHeight / 2 + filterY + ori_img.rows) % ori_img.rows; 
            	//	red += ori_img.at<Vec3b>(imageX)(imageY)[0] * filter[filterX][filterY]; 
            	//	green += ori_img.at<Vec3b>(imageX)(imageY)[1] * filter[filterX][filterY]; 
            	//	blue += ori_img.at<Vec3b>(imageX)(imageY)[2] * filter[filterX][filterY]; 
            		red += ori_img.at<int>(0, imageX,imageY) * filter[filterX][filterY]; 
            		green += ori_img.at<int>(1, imageX,imageY) * filter[filterX][filterY]; 
            		blue += ori_img.at<int>(2, imageX,imageY) * filter[filterX][filterY]; 
				}
			}
			cout << red << endl;
//			out_img.at<char>(x, y, 0) = min(max(int(factor*red + bias), 0), 255); 
//			out_img.at<char>(x, y, 1) = min(max(int(factor*green + bias), 0), 255); 
//			out_img.at<char>(x, y, 2) = min(max(int(factor*blue + bias), 0), 255); 
		}
	}
	
	imshow("out", out_img);
	waitKey();

}
