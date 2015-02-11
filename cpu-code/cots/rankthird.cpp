/*************************************************************************
    > File Name: rankthird.cpp
    > Author: chenrudan
    > Mail: chenrudan123@gmail.com 
    > Created Time: 2014年09月12日 星期五 21时11分34秒
 ************************************************************************/

#include<iostream>
#include<fstream>
#include<map>
#include<vector>

using namespace std;

typedef struct Dot
{
	int origin_pos;
	float value;
}dot;

typedef struct Stimulus
{
	int position;
	float threshold;
}stimulus;

void Merge(dot *source, int start, int mid, int end)  
{  
    int n1 = mid-start+1;  
    int n2 = end-mid; 
    dot *L = new dot[n1+1];  
    dot *R = new dot[n2+1];  
    int i, j, k;  
      
    for (i=0; i<n1; i++){  
        L[i].value = source[start+i].value;
        L[i].origin_pos = source[start+i].origin_pos;
    }  
    for (j=0; j<n2; j++){  
        R[j].value = source[mid+j+1].value;  
        R[j].origin_pos = source[mid+j+1].origin_pos; 
    }  
    L[n1].value = -10000000;  
    R[n2].value = -10000000;  
  
    for (i=0, j=0, k=start; k<=end; k++)  
    {  
        if (L[i].value >= R[j].value)
        {  
            source[k].value = L[i].value;
            source[k].origin_pos = L[i].origin_pos;
            i++;  
        }else{  
            source[k].value = R[j].value;            
			source[k].origin_pos = R[j].origin_pos;
            j++;  
        }  
    }
    delete []L;
    delete []R;
}  
  
void MergeSort(dot *source, int start, int end)  
{  
    if (start < end)  
    {  
        int mid = (start + end)/2;
        MergeSort(source, start, mid);  
        MergeSort(source, mid+1, end);  
        Merge(source, start, mid, end);  
    }  
} 

bool isStimuli(dot *rank, int number, int postive_num)
{
	for(int i = 0; i < number; i++)
	{
		if(rank[i].origin_pos >= postive_num)
			return false;
	}
	return true;
}

float chooseThreshold(dot *all_size_dot, int postive_num, int num, float threshold, float interval, float max_value, float min_value)
{
	int min_cost = 1000000;
	float opt;
	while((threshold >= max_value)||(threshold <= min_value))
	{
		int cost;
		for(int i = 0; i < postive_num; i++)
		{
			if(all_size_dot[i].value <= threshold)
				cost++;
		}
		for(int i = postive_num; i < num; i++)
		{
			if(all_size_dot[i].value >= threshold)
				cost++;
		}
		if(cost < min_cost)
		{
			min_cost = cost;
			opt = threshold;
		}
		threshold += interval;
	}
	return opt;
}

int main()
{
	ifstream fin("../data/test_out3.dat", ios::binary);
	int size = 42;
	int channels = 8;
	int num = 3400;
	int postive_num = 1300;
	int rank_number = 8;
	int length = size*size*channels*num;
	float *buffer = new float[length];
	dot *all_size_dot = new dot[length];	
	fin.read((char *)buffer, length);
	//记录下有可能是stimulti的点
	vector<stimulus> sti;
	for(int i = 0; i < size*size*channels; i++)
	{	
		for(int j = i*num; j < (i+1)*num; j++)
		{
			//保存10万张图的id，看每个节点最敏感的图是什么
			all_size_dot[j].origin_pos = j - i*num;
			all_size_dot[j].value = buffer[j];		
		}
		MergeSort(all_size_dot + i*num, 0, num - 1);
	}
	
//	ofstream fout("../data/rank_dot.bin", ios::binary);
//	fout.write((char*)all_size_dot, length*sizeof(dot));
	//找到某个点排在前面都在postive集
	for(int i = 0; i < 3000; i++)
	{		
		if(isStimuli(all_size_dot + i*num, rank_number, postive_num))
		{
			float max_value = all_size_dot[i*num].value;
			float min_value = all_size_dot[(i+1)*num].value;
			float interval = (max_value - min_value)/20;
			stimulus opt_sti;
			
			opt_sti.threshold = chooseThreshold(all_size_dot + i*num, postive_num, num, min_value, interval, max_value, min_value);
			opt_sti.position = i;
			sti.push_back(opt_sti);
			
			cout << "=========================\n";
			cout << "this point is " << i << endl;
			cout << "=========================\n";
			cout << "the rank_number is " << rank_number << endl;
			cout << "=========================\n"; 	
			cout << "the threshold is " << opt_sti.threshold << endl;	
			cout << "=========================\n"; 
			for(int j = i*num; j < i*num + rank_number; j++)
			{
				cout << all_size_dot[j].origin_pos << ": " <<  all_size_dot[j].value  << endl;					
			}
		
		}
	}
	
	fin.close();
	delete[] buffer;
	delete[] all_size_dot;
	
	return 0;
}

























