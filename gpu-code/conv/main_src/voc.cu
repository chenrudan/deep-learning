///
///  \file conv3.cu
///

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <omp.h>
#include "mpi.h"
#include "train_model.hpp"
#include "convnet.hpp"

using namespace std;


typedef void(*loadFun)(); 

int Param::_minibatch_size = 0;

void managerNode(TrainModel<float> *model){

	cout << "Loading data...\n";

	model->createWBiasForManager();
	cout << "Initialize weight and bias...\n";
	model->createPixelAndLabel();
	cout << "Loading data is done.\n";

	model->initWeightAndBcast();
	model->createMPIDist();
	model->sendAndRecvForManager();
	cout << "CPU number: " << omp_get_num_procs() << endl;  
}

void workerNode(TrainModel<float> *model){

	cout << "Initialize layers...\n";

	model->createLayerForWorker();
	cout << "Initialize layers is done.\n";
	model->createWBiasForWorker();
	cout << "Each process reciving their data...\n";
	model->createPixelAndLabel();
	model->createYDEDYForWorker();
	model->initWeightAndBcast();
	cout << "done2\n";
	model->createMPIDist();
	cout << "done1\n";
	model->train();

}

int main(int argc, char** argv){

	int pid; 
	int num_process;
	int prov;
	MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE, &prov);
	if (prov < MPI_THREAD_MULTIPLE)
	{   
		printf("Error: the MPI library doesn't provide the required thread level\n");
		MPI_Abort(MPI_COMM_WORLD, 0); 
	}   
	MPI_Comm_rank(MPI_COMM_WORLD,&pid);
	MPI_Comm_size(MPI_COMM_WORLD,&num_process);

	if(num_process <= 1){
		printf("Error: process number must bigger than 1\n");
		MPI_Abort(MPI_COMM_WORLD, 0); 
	}

	//检测有几个gpu
	int num_gpu;
	cudaGetDeviceCount(&num_gpu);
	cudaSetDevice(pid % num_gpu);

	cout << num_gpu << endl;
	cout << num_process << endl;	

	TrainModel<float> *voc_model = new TrainModel<float>(pid);
	voc_model->parseNetJson("script/cifar10.json");
	voc_model->parseImgBinary(num_process);


	if(pid == 0){ 
		managerNode(voc_model);
	}   
	else{
		workerNode(voc_model);
	}
	 	
	delete voc_model;
	MPI_Finalize();


	return 0;
}

















