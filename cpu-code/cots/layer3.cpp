/* Filename: layer3.cpp
 * -------------------
 * 
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <time.h>
#include <string>
#include <sstream>
#include "mpi.h"
#include "utils.h"
#include "load.h"
#include "cots.h"
extern "C"{
#include "cblas.h"
}

using namespace std;

void managerNode(int me, Cots layer3);
void workerNode(int me, Cots layer3);

int epoch = 4;
int all_size = 80;
string weight_name;

int main(int argc, char **argv)
{
    int nnode, //进程数，从命令行读取
        me;    //MPI ID
    int layer3_input_size = 70;
    int layer3_input_channels = 8;
    int layer3_filter_channels = 8;
    int layer3_filter_size = 30;
    int layer3_batch_size = 40;
    int layer3_block_size = 2;
    int layer3_step = 2;
    int layer2_thread_num = 21;
    int layer3_process_num = 21*21/layer2_thread_num;
    int layer3_pooling_size = 3;
    float layer3_learning_rate = -0.00001;
    float layer3_learning_rate_alpha = -0.0005;
    float layer3_alpha = 0.01;
    float layer3_momentum = 0;
    float layer3_lambda = 6;
    
    Cots layer3;
    layer3.init(layer3_input_size, layer3_input_channels, layer3_filter_size, layer3_filter_channels, layer3_batch_size, \
             layer3_block_size, layer3_step, layer3_process_num, layer2_thread_num, layer3_pooling_size, layer3_learning_rate, \
             layer3_learning_rate_alpha, layer3_alpha, layer3_momentum, layer3_lambda, "./binaryfile/layer2out_unlabeled.bin", "./binaryfile/layer3out_unlabeled.bin");

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nnode);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    stringstream ss;
    ss << me;
    ss >> weight_name;
    weight_name = "./binaryfile/weight/layer3_" + weight_name + ".bin";

    if(me == 0) 
    {
        cout << "the learning rate is :" << layer3_learning_rate << "\n" << "the layer3_alpha is :" << layer3_alpha << endl; 
        cout << "the all size is : " << all_size << endl;
        managerNode(me, layer3);
    }   
    else
        workerNode(me, layer3);
    MPI_Finalize();
    return 0;
}

void managerNode(int me, Cots layer3)
{
    clock_t t;
    t = clock();
    layer3.trainModel(me, epoch, all_size, false);
    layer3.saveWeight(weight_name);
    t = clock() - t;
    cout << "this train uses " << (float)t/CLOCKS_PER_SEC << "seconds" << endl;
    layer3.clearMemory();
}

void workerNode(int me, Cots layer3)
{
    layer3.trainModel(me, epoch, all_size, false);
    layer3.saveWeight(weight_name);
    layer3.clearMemory();
}



