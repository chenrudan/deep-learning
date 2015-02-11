/* Filename: main.cpp
 * -------------------
 * 这个文件打开一个二进制文件并传入不同线程跟权重相乘
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <time.h>
#include <string>
#include "mpi.h"
#include "utils.h"
#include "load.h"
extern "C"{
#include "cblas.h"
}

using namespace std;

void initInput(int me, struct Pars *pars, int batch_idx, int process_idx);
void workerNode(int me, struct Pars *pars);
void managerNode(int me, struct Pars *pars);
void assignMemory(struct Pars *pars);
void computeH(struct Pars *pars);
void clear(struct Pars *pars);
void computeR(struct Pars *pars);
void computeP(int me, struct Pars *pars, int process_idx);
void buildR(int me, struct Pars *pars, int process_idx, bool ward);
void buildH(int me, struct Pars *pars, int process_idx);
//void buildP(int me, struct Pars *pars, int process_idx, bool ward);
void buildWeight(int me, struct Pars *pars, int process_idx, int batch_idx, bool ward, bool type = false);
void computeDw1(struct Pars *pars);
void computeDw2(struct Pars *pars);
void updateW(int me, struct Pars *pars, int batch_idx);
void trainModel(int me, struct Pars *pars);
void testModel(int me, struct Pars *pars);
void normalizeWeight(struct Pars *pars);
void filterLayer(int me, struct Pars *pars, int batch_idx);
void poolingLayer(int me, struct Pars *pars, int batch_idx);


struct Pars{
    int input_channels;
    int input_size;
    int filter_size;
    int filter_channels;
    int batch_size;
    int block_size;
    int step;
    int process_num;
    int out_size;
    int pooling_size;
    float learning_rate;
    float alpha;

    //zero pass to other thread
    float *block_input;
    //weight for block compute
    float *block_weight;
    float *block_hidden;
    float *block_reconstruct;
    float *block_pooling;
    float *block_lcn;
    float *block_dw1;
    float *block_dw2;
    //r for zero
    float *send_reconstruct;
    float *recieve_reconstruct;
    //hidden for zero
    float *send_hidden;
    float *recieve_hidden;
    //pooling 
    float *send_pooling;
    float *recieve_pooling;
    //lcn
    float *send_lcn;
    float *recieve_lcn;
    //weight for zero
    float *send_weight;
    float *recieve_weight;
    //input for zero
    float *input;
};

int main(int argc, char **argv)
{
    int nnode, //进程数，从命令行读取
        me;    //MPI ID

    struct Pars *pars = new Pars;
    pars->filter_channels = 8;
    pars->filter_size = 10;
    pars->input_channels = 3;
    pars->input_size = 96;
    pars->batch_size = 48;
    pars->block_size = 2;
    pars->step = 2;
    pars->process_num = 44*44/44;
    pars->out_size = pars->input_size - pars->filter_size + 1;
    pars->pooling_size = 3;
    pars->learning_rate = 0.01;
    pars->alpha = 1;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nnode);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);


    if(me == 0)    
        managerNode(me, pars);
    else
        workerNode(me, pars);

    MPI_Finalize();
    clear(pars);
    return 0;
}

void assignMemory(struct Pars *pars)
{
    //1.初始化w
    //weight大小为2*2*8*10*10*3，列为2*2*8，行为10*10*3
    int length = pars->filter_size*pars->filter_size*pars->filter_channels*pars->input_channels*pars->block_size*pars->block_size;
    pars->block_weight = new float[length];
    //2.初始化x
    int block_input_size = pars->batch_size*pars->filter_size*pars->filter_size*pars->input_channels;
    pars->block_input = new float[block_input_size];     //由于输入是间隔产生的，所以需要间隔赋值
    int block_hidden_size = pars->filter_channels*pars->block_size*pars->block_size*pars->batch_size;
    pars->block_hidden = new float[block_hidden_size];
    pars->block_pooling = new float[block_hidden_size]; 
    pars->block_lcn = new float[block_hidden_size];
    pars->block_reconstruct = new float[block_input_size];
    //3.初始化最后的r
    long r_size = pars->input_size*pars->input_size*pars->input_channels*pars->batch_size;
    pars->send_reconstruct = new float[r_size];
    pars->recieve_reconstruct = new float[r_size];
    //4.初始化最后的h
    long h_size = pars->out_size*pars->out_size*pars->filter_channels*pars->batch_size;
    pars->send_hidden = new float[h_size];
    pars->recieve_hidden = new float[h_size];
    pars->send_pooling = new float[h_size];
    pars->recieve_pooling = new float[h_size];
    pars->send_lcn = new float[h_size];
    pars->recieve_lcn = new float[h_size];
}

void initInput(int me, struct Pars *pars, int batch_idx, int process_idx)
{
    //传入每次需要计算的48*10*10*3
    for(int i = 0; i < pars->batch_size; i++)
    {
        for(int j = 0; j < pars->input_channels; j++)
        {
            int start = batch_idx*pars->batch_size*pars->input_channels*pars->input_size*pars->input_size \
                        + i*pars->input_channels*pars->input_size*pars->input_size \
                        + j*pars->input_size*pars->input_size  \
                        + process_idx*pars->step*pars->input_size  \
                        + me*pars->step*pars->input_size;
            Load load;
            float *tmp_input = load.loadPartData("preprocessed.bin", start, \
                    pars->filter_size, pars->input_size, pars->filter_size);
            for(int m = 0; m < pars->filter_size*pars->filter_size; m++)
            {
                //第i列也就是第i个图片获取的数据
                pars->block_input[i*pars->input_channels*pars->filter_size*pars->filter_size \
                    + j*pars->filter_size*pars->filter_size + m] = \
                    tmp_input[m];
            }
        }
    }
    /*
       if(me == 0)
       {
       for(int i = 0; i < pars->batch_size*pars->input_channels*pars->filter_size*pars->filter_size; i++)
       {
       cout << pars->block_input[i] << endl;
       }
       }*/
    cout << "input init success!" << endl;
}

void managerNode(int me, struct Pars *pars)
{
    clock_t t;
    t = clock();
    trainModel(me, pars);
    //save weight
    t = clock() - t;
    time_t rawtime;
    time(&rawtime);
    string dist = ctime(&rawtime);
    dist.append(".bin");
    ofstream fout(dist.c_str(), ios::binary);
    fout.write((char*)pars->recieve_weight, sizeof(float));
    cout << "this train uses " << (float)t/CLOCKS_PER_SEC << "seconds" << endl;
    fout.close();
    delete[] pars->send_weight;
    delete[] pars->recieve_weight;

}

void workerNode(int me, struct Pars *pars)
{
    trainModel(me, pars);
    delete[] pars->send_weight;
    delete[] pars->recieve_weight;
}

void trainModel(int me, struct Pars *pars)
{
    assignMemory(pars);
    int weight_length = pars->out_size*pars->out_size*pars->filter_channels*pars->input_size*pars->input_size*pars->input_channels;
    pars->send_weight = new float[weight_length];
    pars->recieve_weight = new float[weight_length];

    for(int batch_idx = 0; batch_idx < 48/pars->batch_size; batch_idx++)
    {

        filterLayer(me, pars, batch_idx);
        cout << "build success! \n";       
        poolingLayer(me, pars, batch_idx);
        cout << "pooling success! \n";
        updateW(me, pars, batch_idx);
        cout << "update success! \n";
    }
}

void lcnLayer(int me, struct Pars *pars, int batch_idx)
{
	for(int process_idx = 0; process_idx < pars->process_num; process_idx++)
    {
		computeLcn(me, pars, process_idx);
		buildH(me, pars->block_lcn, pars->send_lcn, process_idx);
	}
	long h_size = pars->out_size*pars->out_size*pars->filter_channels*pars->batch_size;
	MPI_Allreduce(pars->send_lcn, pars->recieve_lcn, h_size, MPI_FLOAT, \
                MPI_SUM, MPI_COMM_WORLD);
}

void poolingLayer(int me, struct Pars *pars, int batch_idx)
{
	for(int process_idx = 0; process_idx < pars->process_num; process_idx++)
    {
		computeP(me, pars, process_idx);
		buildH(me, pars->block_pooling, pars->send_pooling, process_idx);
	}
	long h_size = pars->out_size*pars->out_size*pars->filter_channels*pars->batch_size;
	MPI_Allreduce(pars->send_pooling, pars->recieve_pooling, h_size, MPI_FLOAT, \
                MPI_SUM, MPI_COMM_WORLD);
}


void filterLayer(int me, struct Pars *pars, int batch_idx)
{
    // 共44个线程，需要运行44次才能得到88*88个点
    for(int process_idx = 0; process_idx < pars->process_num; process_idx++)
    {
        initInput(me, pars, batch_idx, process_idx);
        buildWeight(me, pars, process_idx, batch_idx, true);
        computeH(pars);
        computeR(pars);           
        buildR(me, pars, process_idx, true);
        buildH(me, pars->block_hidden, pars->send_hidden, process_idx); 
    }
    //进行通信，将r全部加在一起，并得到合成后的r
    long h_size = pars->out_size*pars->out_size*pars->filter_channels*pars->batch_size;
    long r_size = pars->input_size*pars->input_size*pars->input_channels*pars->batch_size;
    MPI_Allreduce(pars->send_reconstruct, pars->recieve_reconstruct, r_size, MPI_FLOAT, \
            MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(pars->send_hidden, pars->recieve_hidden, h_size, MPI_FLOAT, \
            MPI_SUM, MPI_COMM_WORLD);
}

void testModel(int me, struct Pars *pars)
{	
    assignMemory(pars);
    for(int batch_idx = 0; batch_idx < 48/pars->batch_size; batch_idx++)
    {
        filterLayer(me, pars, batch_idx);
    }
}

/* Function: updateW
 * -------------------
 * 更新权重
 */
void updateW(int me, struct Pars *pars, int batch_idx)
{
	int length = pars->filter_size*pars->filter_size*pars->filter_channels*pars->input_channels*pars->block_size*pars->block_size;
    pars->block_dw1 = new float[length]; 
    pars->block_dw2 = new float[length];
	for(int process_idx = 0; process_idx < pars->process_num; process_idx++)
    {       	
        initInput(me, pars, batch_idx, process_idx);
        buildWeight(me, pars, process_idx, batch_idx, false);
        buildR(me, pars, process_idx, false);       
        //计算第一层的dw
        computeDw1(pars);
        //计算第二层的dw
        computeDw2(pars);
        //更新权重
        int length = pars->filter_size*pars->filter_size*pars->filter_channels*pars->input_channels*pars->block_size*pars->block_size;
    	catlas_saxpby(length, 1, pars->block_dw1, 1, 1, pars->block_dw2, 1);
   	 	catlas_saxpby(length, pars->learning_rate, pars->block_dw2, 1, 1, pars->block_weight, 1);
    	normalizeWeight(pars);
        buildWeight(me, pars, process_idx, batch_idx + 1, true);           
    }
    long h_size = pars->out_size*pars->out_size*pars->filter_channels*pars->batch_size;      
    MPI_Allreduce(pars->send_weight, pars->recieve_weight, h_size, MPI_FLOAT, \
           MPI_SUM, MPI_COMM_WORLD);
    delete[] pars->block_dw1;
    delete[] pars->block_dw2;
}

/* Function: computeDw2
 * --------------------
 * 计算pooling层造成的偏导
 */
void computeDw2(struct Pars *pars)
{
    int length = pars->filter_channels*pars->block_size*pars->block_size*pars->batch_size;
    for(int i = 0; i < length; i++)
    {
        pars->block_hidden[i] = pars->block_hidden[i]/pars->block_pooling[i];
    }
    //计算h/p*x'
    int m = pars->block_size*pars->block_size*pars->filter_channels,
        n = pars->input_channels*pars->filter_size*pars->filter_size,
        k = pars->batch_size,
        lda = pars->batch_size,
        ldb = pars->batch_size,
        ldc = pars->input_channels*pars->filter_size*pars->filter_size;
    float alpha = 1,
          beta = 0;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, pars->block_hidden, lda, pars->block_input, ldb, beta, pars->block_dw2, ldc);
}




/* Function: computeDw1
 * --------------------
 * 计算fliter层造成的偏导，h*(r-x)'+ w*(r-x)*x'
 */
void computeDw1(struct Pars *pars)
{
    int block_r_size = pars->batch_size*pars->filter_size*pars->filter_size*pars->input_channels;
    //计算r-x，存在x里面
    catlas_saxpby(block_r_size, -1, pars->block_input, 1, 1, pars->block_reconstruct, 1);
    int block_w_size = pars->filter_size*pars->filter_size*pars->input_channels*pars->block_size*pars->block_size*pars->filter_channels;
    int block_w_r_size = pars->block_size*pars->block_size*pars->filter_channels*pars->batch_size;
    float *block_h_r = new float[block_w_size];
    float *block_w_r = new float[block_w_r_size];
    //计算h*(r-x)'
    int m = pars->block_size*pars->block_size*pars->filter_channels,
        n = pars->input_channels*pars->filter_size*pars->filter_size,
        k = pars->batch_size,
        lda = pars->batch_size,
        ldb = pars->batch_size,
        ldc = pars->input_channels*pars->filter_size*pars->filter_size;
    float alpha = 1,
          beta = 0;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, pars->block_hidden, lda, pars->block_reconstruct, ldb, beta, block_h_r, ldc);
    //计算w*(r-x)
    m = pars->block_size*pars->block_size*pars->filter_channels,
      n = pars->batch_size,
      k = pars->input_channels*pars->filter_size*pars->filter_size,
      lda = pars->input_channels*pars->filter_size*pars->filter_size,
      ldb = pars->batch_size,
      ldc = pars->batch_size;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, pars->block_weight, lda, pars->block_reconstruct, ldb, beta, block_w_r, ldc);
    //计算w*(r-x)*x'
    m = pars->block_size*pars->block_size*pars->filter_channels,
      n = pars->input_channels*pars->filter_size*pars->filter_size,
      k = pars->batch_size,
      lda = pars->batch_size,
      ldb = pars->batch_size,
      ldc = pars->input_channels*pars->filter_size*pars->filter_size;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha, block_w_r, lda, pars->block_input, ldb, beta, pars->block_dw1, ldc);
    //计算h*(r-x)'+ w*(r-x)*x'
    catlas_saxpby(block_w_size, 1, block_h_r, 1, 1, pars->block_dw1, 1);
    delete[] block_h_r;
    delete[] block_w_r;
}


/* Function: computeP
 * ------------------
 * 计算pooling层的值，每次计算2*2的区域，取值则取到3*3，hidden的平方和开方
 */
void computeP(int me, struct Pars *pars, int process_idx)
{
    for(int i = 0; i < pars->batch_size; i++)
    {
        for(int j = 0; j < pars->filter_channels; j++)
        {
            //block的起点
            int start = i*pars->filter_channels*pars->block_size*pars->block_size  \
                        + j*pars->out_size*pars->out_size  \
                        + process_idx*pars->step*pars->out_size \
                        + me*pars->step;
            for(int m = 0; m < pars->block_size; m++)
            {
                for(int n = 0; n < pars->block_size; n++)
                {
                    float sum = 0;
                    //计算周围九个点的平方和
                    for(int k = -1; k < pars->pooling_size - 1; k++)
                    {
                        for(int t = -1; t < pars->pooling_size - 1; t++)
                        {
                            //当点在hidden内时才加到sum中
                            if((me*pars->block_size + n + t > 0)&&(me*pars->block_size + n + t < 88)       \
                                    &&(process_idx*pars->block_size + m + k > 0)&&(process_idx*pars->block_size + m + k < 88))
                                sum += pars->recieve_hidden[start + k*pars->out_size + t]*pars->recieve_hidden[start + k*pars->out_size + t];
                        }
                    }
                    float lambda = 0.1;
                    pars->block_pooling[ i*pars->filter_channels*pars->block_size*pars->block_size  \
                        + j*pars->block_size*pars->block_size  \
                        + m*pars->block_size + n] = lambda*sqrt(sum); 

                    //更新下一个点
                    start += 1;
                }
                start -= pars->block_size;
                start += pars->out_size;
            }
        }
    }
}

/* Function: computeLcn
 * ------------------
 * 计算lcn层的值，每次计算2*2的区域，取值则取到3*3
 */
void computeLcn(int me, struct Pars *pars, int process_idx)
{
    for(int i = 0; i < pars->batch_size; i++)
    {
        for(int j = 0; j < pars->filter_channels; j++)
        {
            //block的起点
            int start = i*pars->filter_channels*pars->block_size*pars->block_size  \
                        + j*pars->out_size*pars->out_size  \
                        + process_idx*pars->step*pars->out_size \
                        + me*pars->step;
            for(int m = 0; m < pars->block_size; m++)
            {
                for(int n = 0; n < pars->block_size; n++)
                {
                    float sum1 = 0;
                    float sum2 = 0;
                    float gaussion[9] = {0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625};
                    //计算周围九个点的平方和
                    for(int k = -1; k < pars->pooling_size - 1; k++)
                    {
                        for(int t = -1; t < pars->pooling_size - 1; t++)
                        {
                            //当点在hidden内时才加到sum中
                            if((me*pars->block_size + n + t > 0)&&(me*pars->block_size + n + t < 88)       \
                                    &&(process_idx*pars->block_size + m + k > 0)&&(process_idx*pars->block_size + m + k < 88))
                                sum1 += gaussion[(k+1)*pars->pooling_size + t + 1]*pars->recieve_pooling[start + k*pars->out_size + t];
                                sum2 += gaussion[(k+1)*pars->pooling_size + t + 1]*pars->recieve_pooling[start + k*pars->out_size + t] \
                                		*pars->recieve_pooling[start + k*pars->out_size + t];
                        }
                    }
                    //subtractive normalizations
                    pars->block_lcn[ i*pars->filter_channels*pars->block_size*pars->block_size  \
                        + j*pars->block_size*pars->block_size  \
                        + m*pars->block_size + n] -= sum1;
                    pars->block_lcn[ i*pars->filter_channels*pars->block_size*pars->block_size  \
                        + j*pars->block_size*pars->block_size  \
                        + m*pars->block_size + n] /= sqrt(sum2) > 0.01 ? sqrt(sum2) : 0.01;
                    //更新下一个点
                    start += 1;
                }
                start -= pars->block_size;
                start += pars->out_size;
            }
        }
    }
}

/* Function: buildP
 * ----------------
 * 将h还原成大的h

void buildP(int me, struct Pars *pars, int process_idx, bool ward)
{
    for(int i = 0; i < pars->batch_size; i++)
    {
        for(int j = 0; j < pars->filter_channels; j++)
        {
            int start = i*pars->filter_channels*pars->out_size*pars->out_size  \
                        + j*pars->out_size*pars->out_size  \
                        + process_idx*pars->step*pars->out_size \
                        + me*pars->step;
            for(int m = 0; m < pars->block_size; m++)
            {
                for(int n = 0; n < pars->block_size; n++)
                {
                    if(ward)
                    {
                        pars->send_pooling[start + n] =  \
                            pars->block_pooling[i*pars->filter_channels*pars->block_size*pars->block_size \
                            + j*pars->block_size*pars->block_size \
                            + m*pars->block_size + n];
                    }
                    else
                    {
                        pars->block_pooling[i*pars->filter_channels*pars->block_size*pars->block_size \
                            + j*pars->block_size*pars->block_size  \
                            + m*pars->block_size + n] = pars->recieve_pooling[start + n];
                    }
                }
                start = start + pars->out_size;
            }
        }
    }
}*/

/* Function: buildH
 * ----------------
 * 将h还原成大的h
 */
void buildH(int me, float *block, float *all, int process_idx)
{
    for(int i = 0; i < pars->batch_size; i++)
    {
        for(int j = 0; j < pars->filter_channels; j++)
        {
            int start = i*pars->filter_channels*pars->out_size*pars->out_size  \
                        + j*pars->out_size*pars->out_size  \
                        + process_idx*pars->step*pars->out_size \
                        + me*pars->step;
            for(int m = 0; m < pars->block_size; m++)
            {
                for(int n = 0; n < pars->block_size; n++)
                {
            //        if(ward)
           //         {
                        all[start + n] =  \
                            block[i*pars->filter_channels*pars->block_size*pars->block_size \
                            + j*pars->block_size*pars->block_size  \
                            + m*pars->block_size + n];
             //       }
            //        else
            //        {
            //            block[i*pars->filter_channels*pars->block_size*pars->block_size \
            //                + j*pars->block_size*pars->block_size  \
            //                + m*pars->block_size + n] = all[start + n];
           //         }
                }
                start = start + pars->out_size;
            }

        }
    }
}


/* Function: buildR
 * ----------------
 * 将r还原成大的r
 */
void buildR(int me, struct Pars *pars, int process_idx, bool ward)
{
    for(int i = 0; i < pars->batch_size; i++)
    {                    
        for(int j = 0; j < pars->input_channels; j++)
        {
            int start = i*pars->input_channels*pars->input_size*pars->input_size \
                        + j*pars->input_size*pars->input_size  \
                        + process_idx*pars->step*pars->input_size   \
                        + me*pars->step;
            for(int m = 0; m < pars->filter_size; m++)
            {
                for(int n = 0; n < pars->filter_size; n++)
                {
                    if(ward)
                    {
                        //将block_r还原到r矩阵
                        pars->send_reconstruct[start + n] = \
                             pars->block_reconstruct[i*pars->input_channels*pars->filter_size*pars->filter_size \
                             + j*pars->filter_size*pars->filter_size + m*pars->filter_size + n];
                    }
                    else
                    {
                        pars->block_reconstruct[i*pars->input_channels*pars->filter_size*pars->filter_size \
                            + j*pars->filter_size*pars->filter_size + m*pars->filter_size + n]  \
                            = pars->recieve_reconstruct[start + n];
                    }
                }
                start = start + pars->input_size;
            }
        }
    }
}


/* Function: computeR
 * ------------------
 * 计算每一个小的r
 */
void computeR(struct Pars *pars)
{
    int m = pars->filter_size*pars->filter_size*pars->input_channels,
        n = pars->batch_size,
        k = pars->filter_channels*pars->block_size*pars->block_size,
        lda = pars->input_channels*pars->filter_size*pars->filter_size,
        ldb = pars->batch_size,
        ldc = pars->batch_size;
    float alpha = 1,
          beta = 0;
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, alpha, pars->block_weight, lda, pars->block_hidden, ldb, beta, pars->block_reconstruct, ldc);
}

/* Function: computeH
 * -------------------
 * 用来计算w*x
 */
void computeH(struct Pars *pars)
{
    int m = pars->filter_channels*pars->block_size*pars->block_size,
        n = pars->batch_size,
        k = pars->filter_size*pars->filter_size*pars->input_channels,
        lda = pars->filter_size*pars->filter_size*pars->input_channels,
        ldb = pars->batch_size,
        ldc = pars->batch_size;
    float alpha = pars->alpha,
          beta = 0;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, pars->block_weight, lda, pars->block_input, ldb, beta, pars->block_hidden, ldc);
}

/* Function：initWeight
 * ---------------------
 * 初始化权重，weight按block来存
 */
void buildWeight(int me, struct Pars *pars, int process_idx, int batch_idx, bool ward, bool type)
{
    if(!type)
    {
        int length = pars->filter_size*pars->filter_size*pars->filter_channels*pars->input_channels*pars->block_size*pars->block_size;
        if((batch_idx != 0)||(!ward))
        {
            int start = process_idx*pars->out_size/pars->step*length \
                        + me*length;
            for(int m = 0; m < length; m++)
            {
                if(ward)
                {
                    pars->send_weight[start + m] = pars->block_weight[m];
                }
                else
                {
                    pars->block_weight[m] = pars->recieve_weight[start + m];
                }
            }
        }
        else
        {
            for(int i = 0; i < length; i++)
            {
                pars->block_weight[i] = RandomNormal();
            }
        }
    }
    //读取文件中的权重值
    else
    {

    }
    cout << "weight init success!" << endl;
    //weight存起来

}

void normalizeWeight(struct Pars *pars)
{
    int length = pars->filter_size*pars->filter_size*pars->filter_channels*pars->input_channels*pars->block_size*pars->block_size;
    float sum = 0;
    for(int i = 0; i < length; i++)
    {
        sum += pars->block_weight[i]*pars->block_weight[i];
    }
    for(int i = 0; i < length; i++)
    {
        pars->block_weight[i] = pars->block_weight[i]/sum;
    }
}

void clear(struct Pars *pars)
{
    delete[] pars->block_input;
    delete[] pars->block_weight;
    delete[] pars->block_hidden;
    delete[] pars->block_reconstruct;
    delete[] pars->block_pooling;
    delete[] pars->block_lcn;
    delete[] pars->send_reconstruct;
    delete[] pars->recieve_reconstruct;
    delete[] pars->send_hidden;
    delete[] pars->recieve_hidden;
    delete[] pars->send_pooling;
    delete[] pars->recieve_pooling;
    delete[] pars->send_lcn;
    delete[] pars->recieve_lcn;
}























