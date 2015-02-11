/*************************************************************************
    > File Name: load_data.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com
    > Created Time: 2014年07月17日 星期四 09时02分33秒
 ************************************************************************/

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include "load.h"

using namespace std;

Load::Load()
{

}

Load::~Load()
{

}

/*
vector<float>* Load::LoadData(string filename)
{
    ifstream fin(filename.c_str(), ios::binary);
    char *buffer;
    fin.seekg(0, fin.end);
    long m = fin.tellg();
    buffer = new char[m];
    fin.seekg (0, fin.beg);
    vector<float> *data = new vector<float>;
    fin.read(buffer, m);
    for(int i = 0; i < m/4; i++)
    {
        unsigned char u_buffer[4];
        u_buffer[0] = buffer[i];
        u_buffer[1] = buffer[i+1];
        u_buffer[2] = buffer[i+2];
        u_buffer[3] = buffer[i+3];
        float tmp;
        fread(&tmp,1,4,fin);
        cout << tmp << endl;
        data->push_back(tmp);
  //      data->push_back(*((float*)u_buffer));
        fin.seekg(4*i, fin.beg);
    }
    fin.close();
    return data;
}*/

vector<float>* Load::LoadData(string filename)
{
    FILE *pf;
    long m;
    vector<float> *data = new vector<float>;
    pf = fopen(filename.c_str(), "rb");
    fseek (pf , 0 , SEEK_END);
    m = ftell (pf);
    rewind(pf);
    for(int i = 0; i < m/4; i++)
    {
        float buffer;
        fread(&buffer,1,4,pf);
        data->push_back(buffer);
        fseek (pf , 0 , 4*i);
    }
    fclose(pf);
    return data;
}















