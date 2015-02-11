/* Filename: layer2.cpp
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

void managerNode(int me, Cots layer2);
void workerNode(int me, Cots layer2);

int epoch = 4;
int all_size = 60000;
string weight_name;

int main(int argc, char **argv)
{
    int nnode, //进程数，从命令行读取
        me;    //MPI ID
    int layer2_input_size = 20;
    int layer2_input_channels = 3;
    int layer2_filter_channels = 3;
    int layer2_filter_size = 10;
    int layer2_batch_size = 40;
    int layer2_block_size = 2;
    int layer2_step = 2;
    int layer2_thread_num = 6;
    int layer2_process_num = 6*6/layer2_thread_num;
    int layer2_pooling_size = 3;
    float layer2_learning_rate = -10;
    float layer2_learning_rate_alpha = -0.01;
    float layer2_alpha = 0.1;
    float layer2_momentum = -0.005;
    float layer2_lambda = 0.1;
    Cots layer2;
    layer2.init(layer2_input_size, layer2_input_channels, layer2_filter_size, layer2_filter_channels, layer2_batch_size, \
             layer2_block_size, layer2_step, layer2_process_num, layer2_thread_num, layer2_pooling_size, layer2_learning_rate, \
             layer2_learning_rate_alpha, layer2_alpha, layer2_momentum, layer2_lambda, "./binaryfile/mnist_layer2_");

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nnode);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    stringstream ss;
    ss << me;
    ss >> weight_name;
    weight_name = "./binaryfile/weight/mnist_layer2_" + weight_name + ".bin";

    if(me == 0)    
    {
        cout << "the learning rate of w is :" << layer2_learning_rate << "\n" << "the lambda is :" << layer2_lambda << endl; 
        cout << "the learning rate of alpha is :" << layer2_learning_rate_alpha << "\n" << "the layer1_alpha is :" << layer2_alpha << endl; 
        cout << "the all size is : " << all_size << "\n"<< "the epoch number is : " << epoch <<endl;
        managerNode(me, layer2);
    }
    else
        workerNode(me, layer2);
    MPI_Finalize();
    return 0;
}

void managerNode(int me, Cots layer2)
{
    clock_t t;
    t = clock();

    layer2.trainModel(me, epoch, all_size, false);
    layer2.saveWeight(weight_name);
    t = clock() - t;
    cout << "this train uses " << (float)t/CLOCKS_PER_SEC << "seconds" << endl;
    layer2.clearMemory();
}

void workerNode(int me, Cots layer2)
{
    layer2.trainModel(me, epoch, all_size, false);
    layer2.saveWeight(weight_name);
    layer2.clearMemory();
}























