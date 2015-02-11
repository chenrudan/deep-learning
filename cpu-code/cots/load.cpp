/*************************************************************************
  > File Name: load_data.cpp
  > Author: chenrudan
  > Mail: chenrudan123@gmail.com
  > Created Time: 2014年07月17日 星期四 09时02分33秒
 ************************************************************************/

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include "load.h"

using namespace std;

Load::Load()
{

}

Load::~Load()
{

}

/*
float* Load::LoadData(string filename)
{
   ifstream fin(filename.c_str(), ios::binary|ios::ate);
   char *buffer;
   float *data;
   fin.seekg(0, fin.end);
   long m = fin.tellg();
   buffer = new char[m];
   data = new float[m];
   fin.seekg (0, fin.beg);
   fin.read(buffer, m);
   for(int i = 0; i < m; i++)
   {
   unsigned char u_buffer = buffer[i];
   data[i] = u_buffer;
   }
   fin.close();
   delete buffer;
   return data;
   }

float* Load::LoadPartData(string filename, long start, int length, int interval, int times)
{
//start表示起始地点，length表示一次读入的长度，times表示读多少次，intervel表示间隔
ifstream fin(filename.c_str(), ios::binary|ios::ate);
char *buffer;
float *data;
buffer = new char[length];
data = new float[length*times];
for(int pos = 0, k = 0; pos < times; pos++)
{
fin.seekg(start + pos*interval, fin.beg);
fin.read(buffer, length);
for(int i = 0; i < length; i++)
{
unsigned char u_buffer = buffer[i];
data[k] = u_buffer;
k++;
}
}
fin.close();
delete buffer;
return data;
}*/

float* Load::loadPartData(string filename, long start, int length, int interval, int times)
{
    FILE *pf;
    float *data = new float[length*times];
    pf = fopen(filename.c_str(), "rb");
    for(int pos = 0, k = 0; pos < times; pos++)
    {
        fseek (pf ,(start + pos*interval)*4, SEEK_SET);
        for(int i = 0; i < length; i++)
        {
            float buffer;
            fread(&buffer,1,4,pf);
            data[k] = buffer;
            k++;
            fseek (pf, 4, SEEK_CUR);
        }
    }
    fclose(pf);
    return data;
}


float* Load::loadData(string filename)
{
    FILE *pf;
    long m;
    pf = fopen(filename.c_str(), "rb");
    fseek (pf , 0 , SEEK_END);
    m = ftell (pf);
    float *data = new float[m/4];
    rewind(pf);
    for(int i = 0; i < m/4; i++)
    {
        float buffer;
        fread(&buffer,1,4,pf);
        data[i] = buffer;
        fseek (pf , 0 , 4*i);
    }
    fclose(pf);
    return data;
}


























