/*************************************************************************
    > File Name: preprocess.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com 
    > Created Time: 2014年09月05日 星期五 14时48分09秒
 ************************************************************************/

#include<iostream>
#include<fstream>
#include<cmath>
using namespace std;

int main()
{
	ifstream fin("../../nej/cpluscode/data/stl10/unlabeled_X.bin", ios::binary);
	ofstream fout("./binaryfile/preprocessed_unlabeled.bin", ios::binary);
	int num = ;
	int channels = 3;
	int size = 96;
	char *buffer = new char[num*size*size*channels];
	float *value  = new float[num*size*size*channels];
	fin.read(buffer, num*size*size*channels);
	for(int i = 0; i < num; i++)
	{
		for(int j = 0; j < channels; j++)
		{
			float sum = 0;
			for(int m = 0; m < size; m++)
			{
				for(int n = 0; n < size; n++)
				{
					sum += buffer[i*channels*size*size + j*size*size + m*size + n];
				}
			}
			float average = sum/(size*size);
			for(int m = 0; m < size; m++)
			{
				for(int n = 0; n < size; n++)
				{
					value[i*channels*size*size + j*size*size + m*size + n] = buffer[i*channels*size*size + j*size*size + m*size + n] - average;
				}
			}
			float square = 0;
			for(int m = 0; m < size; m++)
			{
				for(int n = 0; n < size; n++)
				{
					square += value[i*channels*size*size + j*size*size + m*size + n]*value[i*channels*size*size + j*size*size + m*size + n];
				}
			}
			for(int m = 0; m < size; m++)
			{
				for(int n = 0; n < size; n++)
				{
					value[i*channels*size*size + j*size*size + m*size + n] = value[i*channels*size*size + j*size*size + m*size + n]/sqrt(square);
				}
			}

		}
	}
	fout.write((char *)value, num*size*size*channels*sizeof(float));
	fin.close();
	fout.close();

	return 0;
}
