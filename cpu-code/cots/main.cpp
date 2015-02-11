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
#include "cots.h"
extern "C"{
#include "cblas.h"
}

using namespace std;

void managerNode(int me, Cots cots);
void workerNode(int me, Cots cots);

int main(int argc, char **argv)
{
    int nnode, //进程数，从命令行读取
        me;    //MPI ID
    int layer1_input_size = 96;
    int layer1_input_channels = 3;
    int layer1_filter_channels = 8;
    int layer1_filter_size = 10;
    int layer1_batch_size = 48;
    int layer1_block_size = 2;
    int layer1_step = 2;
    int layer1_process_num = 44*44/44;
    int layer1_pooling_size = 3;
    float layer1_learning_rate = 0.0001;
    float layer1_alpha = 0.01;
    float layer1_momentum = 0.005;
    
    Cots cots(layer1_input_size, layer1_input_channels, layer1_filter_size, layer1_filter_channels, layer1_batch_size, \
             layer1_block_size, layer1_step, layer1_process_num, layer1_pooling_size, layer1_learning_rate, \
             layer1_alpha, layer1_momentum);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nnode);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);


    if(me == 0)    
        managerNode(me, cots);
    else
        workerNode(me, cots);
    MPI_Finalize();
    return 0;
}

void managerNode(int me, Cots cots)
{
    clock_t t;
    t = clock();
    int epoch = 10;
    cots.trainModel(me, epoch);
    t = clock() - t;
    cout << "this train uses " << (float)t/CLOCKS_PER_SEC << "seconds" << endl;
    cots.saveFile();
}

void workerNode(int me, Cots cots)
{
   // trainModel(me, pars);
}


