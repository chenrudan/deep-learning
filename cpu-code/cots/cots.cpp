/*************************************************************************
  > File Name: cots.cpp
  > Author: chenrudan
  > Mail: chenrudan123@gmail.com 
  > Created Time: 2014年08月30日 星期六 20时26分30秒
 ************************************************************************/

#include<iostream>
#include <fstream>
#include <cmath>
#include <time.h>
#include <string>
#include <sstream>
#include <unistd.h>
#include <cstdlib>
#include "mpi.h"
#include "cots.h"
extern "C"{
#include "cblas.h"
}

using namespace std;

Cots::Cots(int input_size, int input_channels, int filter_size, int filter_channels, \
    int batch_size, int block_size, int step, int process_num, int thread_num, \
    int pooling_size, float learning_rate, float learning_rate_alpha, float alpha, \
    float momentum, float lambda, string address){
  init(input_size, input_channels, filter_size, filter_channels, batch_size, block_size,\
      step, process_num, thread_num, pooling_size, learning_rate, learning_rate_alpha, \
      alpha, momentum, lambda, address);
}

Cots::Cots(){}

Cots::~Cots(){}

void Cots::init(int input_size, int input_channels, int filter_size, int filter_channels, \
    int batch_size, int block_size, int step, int process_num, int thread_num, \
    int pooling_size, float learning_rate, float learning_rate_alpha, float alpha, \
    float momentum, float lambda, string address){
  this->_pars = new _Pars;
  this->_pars->filter_channels = filter_channels;
  this->_pars->filter_size = filter_size; 
  this->_pars->input_channels = input_channels;
  this->_pars->input_size = input_size; 
  this->_pars->batch_size = batch_size; 
  this->_pars->block_size = block_size;
  this->_pars->step = step;
  this->_pars->process_num = process_num;
  this->_pars->thread_num = thread_num;
  this->_pars->out_size = this->_pars->input_size - this->_pars->filter_size + 2;
  this->_pars->pooling_size = pooling_size;
  this->_pars->learning_rate = learning_rate;
  this->_pars->alpha = alpha;
  this->_pars->learning_rate_alpha = learning_rate_alpha;
  this->_pars->momentum = momentum;
  this->_address = address;
  this->_lambda = lambda;
  this->_block_sqrt = block_size*block_size;
  this->_input_sqrt = input_size*input_size;
  this->_out_sqrt = this->_pars->out_size*this->_pars->out_size;
  this->_filter_sqrt = filter_size*filter_size;
  this->_pooling_sqrt = pooling_size*pooling_size;
  assignMemory();

}

void Cots::testModel(int me, int epoch, int all_size, bool type){
  initWeight(me, false);

  for(int epoch_idx = 0; epoch_idx < epoch; epoch_idx++){
    for(int batch_idx = 0; batch_idx < all_size/this->_pars->batch_size; batch_idx++){
      if(me == 0){
	if(batch_idx == 0){
	  cout << "=============================\n";
	  cout << "====epoch is " << epoch_idx<< "=====\n";
	}
	cout << "batch_idx is " << batch_idx << "\n";
	//预处理图片
	//                preprocess(this->_input_address, batch_idx, this->_pars->batch_size, this->_pars->input_channels, this->_pars->input_size);
      }
      filterLayer(me, batch_idx);
      poolingLayer(me, batch_idx);
      lcnLayer(me, batch_idx);
      if((me == 0) && (epoch_idx == epoch -1)){
	int length = this->_pars->out_size*this->_pars->out_size \
		     *this->_pars->filter_channels*this->_pars->batch_size;
	subSaveFile("./binaryfile/test_layer1out_unlabeled.bin", length, \
	    this->_pars->receive_lcn, false);
      }

    }
  }
}


void Cots::trainModel(int me, int epoch, int all_size, bool type){
  int weight_length = this->_pars->process_num * this->_pars->filter_channels \
		      * this->_block_sqrt * this->_filter_sqrt  \
		      * this->_pars->input_channels;
  this->_pars->winc = new float[weight_length];
  string savename;

  initWeight(me, true);
  normalizeWeight();

  zeros(this->_pars->winc, weight_length);
  for(int epoch_idx = 0; epoch_idx < epoch; epoch_idx++){
    for(int batch_idx = 0; batch_idx < all_size/this->_pars->batch_size; batch_idx++){
      int r_size = this->_input_sqrt * this->_pars->input_channels \
		   * this->_pars->batch_size;
      int h_size = this->_out_sqrt * this->_pars->filter_channels \
		   * this->_pars->batch_size;
      if(me == 0){
	cout << "=============================\n";
	cout << "====epoch is " << epoch_idx <<  \
	  "   batch_idx is " << batch_idx << "=====\n";
	//预处理图片
	preprocess( batch_idx, this->_pars->batch_size, this->_pars->input_channels, \
	    this->_pars->input_size, this->_pars->input, type);
      }
      MPI_Bcast(this->_pars->input, r_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

      filterLayer(me, batch_idx);
      poolingLayer(me, batch_idx);
      lcnLayer(me, batch_idx);
      updateW(me, batch_idx);
      normalizeWeight();

      if(me == 0){
	updateAlpha();
	this->_pars->alpha -= this->_pars->learning_rate_alpha*this->_delta_alpha;
      }
      MPI_Bcast(&this->_pars->alpha, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

      if(me == 0 ){
	float cost = 0;
	for(int i = 0 ; i < r_size; i++){
	  cost += (this->_pars->receive_reconstruct[i] - this->_pars->input[i]) \ 
	    *(this->_pars->receive_reconstruct[i] - this->_pars->input[i]);
	}
	float cost1 = 0;
	for(int i = 0 ; i < h_size; i++){
	  cost1 += this->_pars->receive_pooling[i];
	}
	cout << "the delta alpha is " << this->_pars->learning_rate_alpha*this->_delta_alpha \
	  << "       "  << "the alpha is     " <<  this->_pars->alpha<< endl;            
	cout << "============================================================================================================\n";
	cout << "============================================================================================================\n";
	cout << "cost of reconstruct is  " << cost << "  cost of pooling is  " \
	  << cost1*this->_lambda <<"     total cost is " \
	  << cost + cost1*this->_lambda << endl;
	cout << "============================================================================================================\n";
	cout << "============================================================================================================\n";

	cout << "==========input=============\n";
	for(int i = 0; i< 10; i++){
	  cout << this->_pars->input[i] << "\t";
	}
	cout << endl;
	cout << "==========reconstruct=============\n";
	for(int i = 0; i< 10; i++){
	  cout << this->_pars->receive_reconstruct[i]<< "\t";
	}
	cout << endl;
	cout << "==========hidden=============\n";
	for(int i = 0; i< 10; i++){
	  cout<<this->_pars->receive_hidden[i] << "\t";
	}
	cout << endl;
	cout << "==========pooling=============\n";
	for(int i = 0; i< 10; i++){
	  cout<< this->_pars->receive_pooling[i] << "\t";
	}
	cout << endl;
	cout << "==========lcn=============\n";
	for(int i = 0; i< 10; i++){
	  cout<< this->_pars->receive_lcn[i]<< "\t";
	}
	cout << endl;  

	if(epoch_idx == epoch - 1){
	  int length = this->_input_sqrt * this->_pars->input_channels \
		       * this->_pars->batch_size;
	  savename = this->_address + "reconstruct.bin";
	  subSaveFile(savename, length, this->_pars->receive_reconstruct, true);
	  length = this->_out_sqrt * this->_pars->filter_channels \
		   * this->_pars->batch_size;
	  savename = this->_address + "hidden.bin";
	  subSaveFile(savename, length, this->_pars->receive_hidden, false);
	  savename = this->_address + "pooling.bin";
	  subSaveFile(savename, length, this->_pars->receive_pooling, false);
	  savename = this->_address + "out.bin";
	  subSaveFile(savename, length, this->_pars->receive_lcn, false);
	}
      }
    }
    if((epoch_idx + 1) % 10 == 0){
      const int weight_all_length = weight_length*this->_pars->thread_num;
      float *all_w = new float[weight_all_length];
      MPI_Gather(this->_pars->weight, weight_length, MPI_FLOAT, all_w, weight_length, \
	  MPI_FLOAT, 0, MPI_COMM_WORLD); 
      if(me == 0){
	stringstream ss2;
	ss2 << epoch_idx;
	savename = this->_address  + ss2.str() +  "_w.bin";
	subSaveFile(savename, weight_all_length, all_w, true);
      }
      delete[] all_w;
    }
  }
  delete[] this->_pars->winc;
}

void Cots::preprocess(int batch_idx, int num, int channels, \
    int size, float *input, bool type){
  string filename = this->_address + "in.bin";
  ifstream fin(filename.c_str(), ios::binary);
  int size_sqrt = size*size;
  if(!fin){
    cout << "the input file cannot be open! \n";
    return;
  }
  if(type){
    char *buffer = new char[num*size*size*channels];
    int pos = batch_idx*num*channels*size_sqrt;
    fin.seekg(pos, fin.beg);
    fin.read(buffer, num*size_sqrt*channels);
    for(int i = 0; i < num; i++){
      for(int j = 0; j < channels; j++){
	float sum = 0;
	for(int m = 0; m < size; m++){
	  for(int n = 0; n < size; n++){
	    //		                sum += buffer[i*channels*size*size + j*size*size + m*size + n];
	    int tmp_pos = i*channels*size_sqrt + j*size_sqrt \
			  + m*size + n;
	    unsigned char tmp = buffer[tmp_pos];
	    sum += tmp;
	  }
	}
	float average = sum/size_sqrt;
	for(int m = 0; m < size; m++){
	  for(int n = 0; n < size; n++){
	    int tmp_pos = i*channels*size_sqrt + j*size_sqrt \
			  + m*size + n;
	    unsigned char tmp = buffer[tmp_pos];
	    input[tmp_pos] = tmp - average;
	  }
	}
	float square = 0;
	for(int m = 0; m < size; m++){
	  for(int n = 0; n < size; n++){
	    int tmp_pos = i*channels*size_sqrt + j*size_sqrt \
			  + m*size + n;
	    square += input[tmp_pos]*input[tmp_pos];
	  }
	}
	for(int m = 0; m < size; m++){
	  for(int n = 0; n < size; n++){
	    int tmp_pos = i*channels*size_sqrt + j*size_sqrt \
			  + m*size + n;
	    input[tmp_pos] = input[tmp_pos]/sqrt(square);
	  }
	}

      }
    }
    delete[] buffer;
  }
  else{
    int batch_length = num*size_sqrt*channels;
    float *buffer = new float[batch_length];
    int pos = batch_idx*batch_length*sizeof(float);
    fin.seekg(pos, fin.beg);
    fin.read((char *)buffer, batch_length*sizeof(float));
    for(int i = 0; i < num; i++){
      for(int j = 0; j < channels; j++){

	float sum = 0;
	for(int m = 0; m < size; m++){
	  for(int n = 0; n < size; n++){
	    int tmp_pos = i*channels*size_sqrt \
			  + j*size_sqrt + m*size + n;
	    sum += buffer[tmp_pos];
	  }
	}
	float average = (float)sum/size_sqrt;
	for(int m = 0; m < size; m++){
	  for(int n = 0; n < size; n++){
	    int tmp_pos = i*channels*size_sqrt \
			  + j*size_sqrt + m*size + n;
	    input[tmp_pos] = buffer[tmp_pos] - average;
	  }
	}
	float square = 0;
	for(int m = 0; m < size; m++){
	  for(int n = 0; n < size; n++){
	    int tmp_pos = i*channels*size_sqrt \
			  + j*size_sqrt + m*size + n;
	    square += input[tmp_pos]*input[tmp_pos];
	  }
	}
	for(int m = 0; m < size; m++){
	  for(int n = 0; n < size; n++){
	    int tmp_pos = i*channels*size_sqrt \
			  + j*size_sqrt + m*size + n;
	    input[tmp_pos] = input[tmp_pos]/sqrt(square);
	  }
	}

      }
    }
    delete[] buffer;
  }
  fin.close();
}

void Cots::initWeight(int me, bool type){
  int length = this->_pars->process_num * this->_filter_sqrt \
	       * this->_pars->filter_channels * this->_block_sqrt \
	       * this->_pars->input_channels;
  if(type){
    for(int i = 0; i < length; i++){
      float tmp = (rand()%2000 - 1000)/1000.0;
      this->_pars->weight[i] = tmp;
    }
  }
  else{
    //读取文件中weight
    string weight_name;
    stringstream ss;
    ss << me;
    ss >> weight_name;
    weight_name = "./binary/weight/layer1_" + weight_name + ".bin";
    ifstream fin(weight_name.c_str(), ios::binary);
    fin.read((char *)this->_pars->weight, length*sizeof(float));       
  }
}

void Cots::filterLayer(int me, int batch_idx){
  int r_size = this->_input_sqrt * this->_pars->input_channels \
	       * this->_pars->batch_size;
  int h_size = this->_out_sqrt * this->_pars->filter_channels \
	       * this->_pars->batch_size;
  zeros(this->_pars->send_reconstruct, r_size);
  zeros(this->_pars->send_hidden, h_size);
  zeros(this->_pars->receive_reconstruct, r_size);
  zeros(this->_pars->receive_hidden, h_size);
  for(int process_idx = 0; process_idx < this->_pars->process_num; process_idx++){
    buildR(me, this->_pars->block_input, this->_pars->input, \
	process_idx, false);
    computeH(process_idx);
    computeR(process_idx);           
    buildH(me, this->_pars->block_hidden, this->_pars->send_hidden, \
	process_idx, true); 
    buildR(me, this->_pars->block_reconstruct, \
	this->_pars->send_reconstruct, process_idx, true);
  }
  //进行通信，将r全部加在一起，并得到合成后的r
  MPI_Allreduce(this->_pars->send_reconstruct, this->_pars->receive_reconstruct, \
      r_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(this->_pars->send_hidden, this->_pars->receive_hidden, \
      h_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}

void Cots::poolingLayer(int me, int batch_idx){
  int h_size = this->_out_sqrt * this->_pars->filter_channels \
	       * this->_pars->batch_size;
  zeros(this->_pars->send_pooling, h_size); 
  zeros(this->_pars->receive_pooling, h_size); 
  for(int process_idx = 0; process_idx < this->_pars->process_num; process_idx++){
    computeP(me, process_idx, 0);
    buildH(me, this->_pars->block_pooling, \
	this->_pars->send_pooling, process_idx, true);
  }
  MPI_Allreduce(this->_pars->send_pooling, this->_pars->receive_pooling, \
      h_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}

void Cots::lcnLayer(int me, int batch_idx){
  int h_size = this->_pars->out_size*this->_pars->out_size \
	       *this->_pars->filter_channels*this->_pars->batch_size;
  zeros(this->_pars->send_lcn, h_size);
  zeros(this->_pars->receive_lcn, h_size);
  for(int process_idx = 0; process_idx < this->_pars->process_num; process_idx++){   
    computeLcn(me, process_idx, true);
    buildH(me, this->_pars->block_lcn, this->_pars->send_lcn, \
	process_idx, true);
  } 
  MPI_Allreduce(this->_pars->send_lcn, this->_pars->receive_lcn, \
      h_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

  zeros(this->_pars->send_lcn, h_size);
  for(int process_idx = 0; process_idx < this->_pars->process_num; process_idx++){   
    computeLcn(me, process_idx, false);
    buildH(me, this->_pars->block_lcn, this->_pars->send_lcn, \
	process_idx, true);
  }   
  MPI_Allreduce(this->_pars->send_lcn, this->_pars->receive_lcn, \
      h_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}

void Cots::computeH(int process_idx){
  int m = this->_pars->filter_channels * this->_block_sqrt,
      n = this->_pars->batch_size,
      k = this->_filter_sqrt * this->_pars->input_channels;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, \
      this->_pars->alpha, this->_pars->normalize_weight + m*k*process_idx, \
      k, this->_pars->block_input, n, 0, this->_pars->block_hidden, n);
}

void Cots::buildH(int me, float *block, float *all, int process_idx, bool ward){
  int id = process_idx * this->_pars->thread_num + me;
  int row_block = id / (this->_pars->out_size / this->_pars->step);
  int col_block = id % (this->_pars->out_size / this->_pars->step);
  int block_start = row_block * this->_pars->step * this->_pars->out_size \
		    + col_block * this->_pars->step;

  for(int i = 0; i < this->_pars->batch_size; i++){
    for(int j = 0; j < this->_pars->filter_channels; j++){
      int start = block_start + i * this->_pars->filter_channels \
		  * this->_out_sqrt + j * this->_out_sqrt;
      int pos_in_batch = j * this->_block_sqrt \
			 * this->_pars->batch_size + i;

      for(int m = 0; m < this->_pars->block_size; m++){
	for(int n = 0; n < this->_pars->block_size; n++){
	  int pos = pos_in_batch + (m*this->_pars->block_size + n) \
		    *this->_pars->batch_size;
	  int all_pos = start + m*this->_pars->out_size + n;
	  if(ward){
	    all[all_pos] = block[pos];
	  }
	  else{
	    block[pos] = all[all_pos];
	  }
	}
      }
    }
  }
}

void Cots::computeR(int process_idx){
  int m = this->_filter_sqrt * this->_pars->input_channels,
      n = this->_pars->batch_size,
      k = this->_pars->filter_channels * this->_block_sqrt;
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, \
      1, this->_pars->normalize_weight + m*k*process_idx, \
      m, this->_pars->block_hidden, n, 0, \
      this->_pars->block_reconstruct, n);
}

void Cots::buildR(int me, float *block, float *all, int process_idx, bool ward)
{
  int id = process_idx * this->_pars->thread_num + me;
  int row_block = id / (this->_pars->out_size / this->_pars->step);
  int col_block = id % (this->_pars->out_size / this->_pars->step);
  int block_start = row_block * this->_pars->step * this->_pars->input_size \
		    + col_block * this->_pars->step;

  for(int i = 0; i < this->_pars->batch_size; i++){                    
    for(int j = 0; j < this->_pars->input_channels; j++){
      int start = block_start + i * this->_pars->input_channels \
		  * this->_input_sqrt + j * this->_input_sqrt;  
      int pos_in_batch = j * this->_filter_sqrt \
			 * this->_pars->batch_size + i;

      for(int m = 0; m < this->_pars->filter_size; m++){
	for(int n = 0; n < this->_pars->filter_size; n++){
	  int pos = pos_in_batch + (m*this->_pars->filter_size + n) \
		    *this->_pars->batch_size;
	  int all_pos = start + m*this->_pars->input_size + n;
	  if(ward){
	    //将block_r还原到r矩阵
	    all[all_pos] += block[pos];
	  }
	  else{
	    block[pos] = all[all_pos];
	  }
	}
      }
    }
  }
}

void Cots::normalizeWeight(){
  //将所有的weight都normalize
  int length1 = this->_pars->process_num * this->_block_sqrt \
		* this->_pars->filter_channels;
  int length2 = this->_filter_sqrt * this->_pars->input_channels;
  for(int m = 0; m < length1; m++){
    float sum = 0.0001;
    int process_pos = m*length2;
    for(int i = 0; i < length2; i++){
      int w_pos = process_pos + i;
      sum += this->_pars->weight[w_pos]*this->_pars->weight[w_pos];
    }
    sum = sqrt(sum);
    for(int i = 0; i < length2; i++){
      int w_pos = process_pos + i;
      this->_pars->normalize_weight[w_pos] \
	= this->_pars->weight[w_pos]/sum;
    }
  }
}

void Cots::inverseProjWeight(float *graident, float *origin_weight, \
    float *project_weight) {
  int length1 = this->_pars->filter_channels * this->_block_sqrt;
  int length2 = this->_filter_sqrt * this->_pars->input_channels;
  for(int m = 0; m < length1; m++){
    float sum = 0.0001;
    float sum1 = 0;
    int process_pos = m * length2;
    for(int i = 0; i < length2; i++){
      int w_pos = process_pos + i;
      sum += origin_weight[w_pos] * origin_weight[w_pos];
      sum1 += origin_weight[w_pos] * graident[w_pos];
    }
    for(int i = 0; i < length2; i++){
      int w_pos = process_pos + i;
      graident[w_pos] = graident[w_pos]/sqrt(sum) \
			- project_weight[w_pos]*sum1/sum;
    }
  }
}

void Cots::computeP(int me, int process_idx, int type){
  int boundary = this->_pars->out_size / this->_pars->block_size;
  int id = process_idx * this->_pars->thread_num + me;
  int row_block = id / boundary;
  int col_block = id % boundary;
  int block_start = row_block*this->_pars->step*this->_pars->out_size \
		    + col_block*this->_pars->step;
  int a[18] = {-1,-1,-1,0,-1,1,0,-1,0,0,0,1,1,-1,1,0,1,1};
  for(int i = 0; i < this->_pars->batch_size; i++){
    for(int j = 0; j < this->_pars->filter_channels; j++){
      //block的起点
      int start = block_start + i*this->_pars->filter_channels \
		  * this->_out_sqrt + j*this->_out_sqrt;
      int pos_in_batch = j * this->_block_sqrt \
			 * this->_pars->batch_size + i;

      for(int m = 0; m < this->_pars->block_size; m++){
	for(int n = 0; n < this->_pars->block_size; n++){
	  float sum = 0;
	  //计算周围九个点的平方和
	  int all_pos = start + m * this->_pars->out_size + n;
	  for(int k = 0; k < 9; k++){
	    int row = row_block + m + a[2 * k];
	    int col = col_block + n + a[2 * k + 1];
	    //当点在hidden内时才加到sum中
	    if((col >= 0)&&(col <= boundary)&&(row >= 0)&&(row <= boundary)){
	      int around_pos = all_pos + a[2 * k]*this->_pars->out_size \
			       + a[2 * k + 1]; 
	      //type, 0代表计算pooling， 1代表计算delta_w, 2代表计算delta_alpha
	      switch(type){
		case 0:{
			 sum += this->_pars->receive_hidden[around_pos] \
				* this->_pars->receive_hidden[around_pos];
			 break;
		       }
		case 1:{
			 if(this->_pars->receive_pooling[around_pos] != 0){
			   sum += this->_pars->receive_hidden[all_pos] \
				  / this->_pars->receive_pooling[around_pos];
			 }
			 break;
		       }

	      }
	    }
	  }
	  int pos = pos_in_batch + (m * this->_pars->block_size + n) \
		    * this->_pars->batch_size;
	  switch(type){
	    case 0:{
		     this->_pars->block_pooling[pos] = sqrt(sum); 
		     break;
		   }
	    case 1:{
		     this->_pars->block_pooling[pos] = sum; 
		     break;
		   }
	  }
	}
      }
    }
  }
}

void Cots::computeLcn(int me, int process_idx, bool type){
  int id = process_idx*this->_pars->thread_num + me;
  int boundary = this->_pars->out_size/this->_pars->block_size;
  int row_block = id/boundary;
  int col_block = id%boundary;
  int block_start = row_block*this->_pars->step*this->_pars->out_size \
		    + col_block*this->_pars->step;
  float gaussion[9] = {0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625};
  //type表示是计算减法，还是计算除法
  for(int i = 0; i < this->_pars->batch_size; i++){
    float *sum1 = new float[this->_block_sqrt];
    float *sum2 = new float[this->_block_sqrt];       
    zeros(sum1, this->_block_sqrt);
    zeros(sum2, this->_block_sqrt);
    for(int j = 0; j < this->_pars->filter_channels; j++){
      //block的起点
      int start = block_start + i * this->_pars->filter_channels \
		  * this->_out_sqrt + j*this->_out_sqrt;
      for(int m = 0; m < this->_pars->block_size; m++){
	for(int n = 0; n < this->_pars->block_size; n++){                    
	  int sum_pos = m*this->_pars->block_size + n;
	  int all_pos = start + m*this->_pars->out_size + n;
	  //计算八张图周围九个点的平方和
	  for(int k = 0; k < 9; k++){
	    int row = row_block + m + gaussion[2*k];
	    int col = col_block + n + gaussion[2*k + 1];
	    //当点在hidden内时才加到sum中
	    if((col >= 0)&&(col <= boundary) \
		&&(row >= 0)&&(row <= boundary)){
	      int around_pos = all_pos + gaussion[2*k]*this->_pars->out_size \
			       + gaussion[2*k + 1]; 
	      if(type)
		sum1[sum_pos] += gaussion[k]*this->_pars->receive_pooling[around_pos];
	      else{
		sum2[sum_pos] += gaussion[k]*this->_pars->receive_lcn[around_pos] \
				 *this->_pars->receive_lcn[around_pos]; 
	      }
	    }
	  }
	}
      }
    }
    for(int j = 0; j < this->_pars->filter_channels; j++){
      int start = block_start + i * this->_pars->filter_channels \
		  * this->_out_sqrt + j*this->_out_sqrt;
      int pos_in_batch = j * this->_block_sqrt \
			 * this->_pars->batch_size + i;

      for(int m = 0; m < this->_pars->block_size; m++){
	for(int n = 0; n < this->_pars->block_size; n++){       	
	  //subtractive normalizations
	  int pos = pos_in_batch + (m*this->_pars->block_size + n) \
		    *this->_pars->batch_size;
	  int all_pos = start + m*this->_pars->out_size + n;
	  int sum_pos = m*this->_pars->block_size + n;
	  if(type){
	    this->_pars->block_lcn[pos] = this->_pars->receive_pooling[all_pos] \
					  - sum1[sum_pos]/this->_pars->filter_channels;

	  }
	  else{
	    this->_pars->block_lcn[pos] = this->_pars->receive_lcn[all_pos] \
					  /(sqrt(sum2[sum_pos]) > 0.01 ? \
					      sqrt(sum2[sum_pos]) : 0.01);


	  }//更新下一个点
	}
      }
    }
    delete[] sum1;
    delete[] sum2;
  }
}

void Cots::updateW(int me, int batch_idx){
  int length = this->_filter_sqrt * this->_pars->filter_channels \
	       * this->_pars->input_channels * this->_block_sqrt;
  int r_length = this->_pars->batch_size * this->_filter_sqrt \
		 * this->_pars->input_channels;
  int h_length = this->_pars->filter_channels * this->_block_sqrt \
		 * this->_pars->batch_size;
  float *block_dw1 = new float[length]; 
  float *block_dw2 = new float[length];
  for(int process_idx = 0; process_idx < this->_pars->process_num; process_idx++){       	
    zeros(block_dw1, length);
    zeros(block_dw2, length);
    zeros(this->_pars->block_input, r_length);
    zeros(this->_pars->block_hidden, h_length);
    zeros(this->_pars->block_reconstruct, r_length);
    zeros(this->_pars->block_pooling, h_length);

    buildR(me, this->_pars->block_input, this->_pars->input, \
	process_idx, false);        
    buildH(me, this->_pars->block_hidden, this->_pars->receive_hidden, \
	process_idx, false);
    buildR(me, this->_pars->block_reconstruct, \
	this->_pars->receive_reconstruct, process_idx, false);

    /*        if(me == 0){
	      cout << "\n==============x,r============\n";
	      for(int i = 0; i < 10; i++){
	      cout << this->_pars->block_input[i] << \
	      ":" << this->_pars->block_reconstruct[i] << ":" \
	      << this->_pars->block_reconstruct[i] - this->_pars->block_input[i];
	      }
	      cout << "\n==============x,r============\n";
	      for(int i = 0; i < h_length; i++){
	      cout << this->_pars->block_hidden[i] << "\t";
	      }
	      cout << endl;
	      }*/

    //计算第一层的dw
    computeDw1(block_dw1, process_idx);

    //计算第二层的dw
    computeP(me, process_idx, 1);
    computeDw2(block_dw2);

    //更新权重
    if(me == 0 && process_idx == 0){
      cout << "=========delta_w1, delta_w2=========\n";
      for(int i = 0; i < 100; i++)
      {
	cout << block_dw1[i]  << ":" << block_dw2[i] << "\t";
      }
      cout << endl;
    }

    catlas_saxpby(length, 2, block_dw1, 1, 1, block_dw2, 1);

    //将dw反向投影回原平面
    inverseProjWeight(block_dw2, this->_pars->weight + process_idx*length, \
	this->_pars->normalize_weight + process_idx*length);
    catlas_saxpby(length, this->_pars->momentum, \
	this->_pars->winc + process_idx*length, \
	1, this->_pars->learning_rate, block_dw2, 1);
    catlas_saxpby(length, -1, block_dw2, 1, 1, \
	this->_pars->weight + process_idx*length, 1);
    for(int i = 0; i < length; i++){
      int pos = process_idx*length + i;
      this->_pars->winc[pos] = block_dw2[i];
    }
  }
  if(me == 0){
    cout << "=========weight=========\n";
    for(int i = 0; i < 100; i++){
      cout << i << ":" << this->_pars->weight[i] << "\t";
    }
    cout << endl;
    cout << "=========delta_weight=========\n";
    for(int i = 0; i < 100; i++){
      cout << i << ":" << block_dw2[i] << "\t";
    }
    cout << endl;
  }

  delete[] block_dw1;
  delete[] block_dw2;
}

void Cots::updateAlpha(){
  float sum = 0;
  this->_delta_alpha = 0;
  int r_all_length = this->_input_sqrt * this->_pars->input_channels \
		     * this->_pars->batch_size;
  for(int i = 0; i < r_all_length; i++){
    sum += 2*(this->_pars->receive_reconstruct[i] - \
	this->_pars->input[i])*this->_pars->receive_reconstruct[i];
  }
  this->_delta_alpha += sum/(this->_pars->alpha*this->_pars->batch_size);
  cout << "the delta alpha of reconstruct is : " 
    << this->_delta_alpha << "     ";
  sum = 0;
  int h_length = this->_pars->filter_channels*this->_out_sqrt \
		 * this->_pars->batch_size;
  for(int i = 0; i < h_length; i++){
    sum += this->_pars->receive_pooling[i];
  }
  this->_delta_alpha += this->_lambda*sum \
			/(this->_pars->alpha*this->_pars->batch_size);
  cout << "the delta alpha of pooling is : " 
    << this->_lambda*sum/(this->_pars->alpha*this->_pars->batch_size) << "\n";
}

void Cots::computeDw1(float *block_dw1, int process_idx){
  int m = this->_block_sqrt * this->_pars->filter_channels,
      n = this->_pars->input_channels * this->_filter_sqrt,
      k = this->_pars->batch_size;

  int block_r_size = n * k;
  //计算r-x，存在r里面
  catlas_saxpby(block_r_size, -1, this->_pars->block_input, 1, \
      1, this->_pars->block_reconstruct, 1);
  int block_w_r_size = m * k;
  float *block_w_r = new float[block_w_r_size];
  zeros(block_w_r, block_w_r_size);
  //计算h*(r-x)'
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, \
      1, this->_pars->block_hidden, k, \
      this->_pars->block_reconstruct, k, 0, block_dw1, n);
  //计算w*(r-x)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, k, n, \
      1, this->_pars->normalize_weight + m * n * process_idx, \
      n, this->_pars->block_reconstruct, k, 0, block_w_r, k);
  //计算w*(r-x)*x'
  //计算h*(r-x)'+ alpha*w*(r-x)*x'
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, \
      this->_pars->alpha, block_w_r, k, this->_pars->block_input, \
      k, 1, block_dw1, n);
  delete[] block_w_r;
}

void Cots::computeDw2(float *block_dw2){
  //计算alpha*h/p*x'
  int m = this->_block_sqrt * this->_pars->filter_channels,
      n = this->_pars->input_channels*this->_filter_sqrt,
      k = this->_pars->batch_size;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, \
      this->_lambda*this->_pars->alpha, this->_pars->block_pooling, \
      k, this->_pars->block_input, k, 0, block_dw2, n);
}

void Cots::subSaveFile(string filename, int length, float *data, bool type){
  ofstream fout;
  if(type)
    fout.open(filename.c_str(), ios::binary);
  else
    fout.open(filename.c_str(), ios::binary|ios::app|ios::out);
  if(!fout){
    cout << "the output file " + filename + " cannot be open!\n";
    return;
  }
  fout.write((char*)data, length*sizeof(float));
  fout.close();

}

void Cots::zeros(float *all, int length){
  for(int i = 0; i < length; i++){
    all[i] = 0;
  }
}

void Cots::assignMemory(){
  //1.初始化w
  //weight大小为2*2*8*10*10*3，列为2*2*8，行为10*10*3
  int weight_length = this->_pars->process_num * this->_filter_sqrt \
		      * this->_pars->input_channels * this->_block_sqrt \
		      * this->_pars->filter_channels;
  this->_pars->weight = new float[weight_length];
  this->_pars->normalize_weight = new float[weight_length];
  zeros(this->_pars->weight, weight_length);
  zeros(this->_pars->normalize_weight, weight_length);
  //2.初始化x
  int block_input_size = this->_pars->batch_size * this->_filter_sqrt \
			 * this->_pars->input_channels;
  this->_pars->block_input = new float[block_input_size];     
  this->_pars->block_reconstruct = new float[block_input_size];
  zeros(this->_pars->block_input, block_input_size);
  zeros(this->_pars->block_reconstruct, block_input_size);

  int block_hidden_size = this->_pars->filter_channels * this->_block_sqrt \
			  * this->_pars->batch_size;
  this->_pars->block_hidden = new float[block_hidden_size];
  this->_pars->block_pooling = new float[block_hidden_size]; 
  this->_pars->block_lcn = new float[block_hidden_size];
  zeros(this->_pars->block_hidden, block_hidden_size);
  zeros(this->_pars->block_pooling, block_hidden_size);
  zeros(this->_pars->block_lcn, block_hidden_size);
  //3.初始化最后的r
  int r_size = this->_input_sqrt * this->_pars->input_channels \
	       * this->_pars->batch_size;
  this->_pars->input = new float[r_size];
  this->_pars->send_reconstruct = new float[r_size];
  this->_pars->receive_reconstruct = new float[r_size];
  zeros(this->_pars->input, r_size);
  //4.初始化最后的h
  int h_size = this->_out_sqrt * this->_pars->filter_channels \
	       * this->_pars->batch_size;
  this->_pars->send_hidden = new float[h_size];
  this->_pars->receive_hidden = new float[h_size];
  this->_pars->send_pooling = new float[h_size];
  this->_pars->receive_pooling = new float[h_size];
  this->_pars->send_lcn = new float[h_size];
  this->_pars->receive_lcn = new float[h_size];
}

void Cots::clearMemory(){
  delete[] this->_pars->block_input;
  delete[] this->_pars->block_hidden;
  delete[] this->_pars->block_reconstruct;
  delete[] this->_pars->block_pooling;
  delete[] this->_pars->block_lcn;
  delete[] this->_pars->input;
  delete[] this->_pars->send_reconstruct;
  delete[] this->_pars->receive_reconstruct;
  delete[] this->_pars->send_hidden;
  delete[] this->_pars->receive_hidden;
  delete[] this->_pars->send_pooling;
  delete[] this->_pars->receive_pooling;
  delete[] this->_pars->send_lcn;
  delete[] this->_pars->receive_lcn;
  delete[] this->_pars->weight;
  delete[] this->_pars->normalize_weight;
  delete this->_pars;
}








