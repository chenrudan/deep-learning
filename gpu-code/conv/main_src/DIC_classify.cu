///
///  \file conv3.cu
///

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <omp.h>
#include "mpi.h"
#include "train_classification.hpp"
#include "convnet.hpp"

using namespace std;


int Param::_minibatch_size = 0;

void managerNode(TrainClassification<float> *model){

	cout << "Loading data...\n";
	model->createWBiasForManager();
	cout << "Initialize weight and bias...\n";
	model->createPixelAndLabel();
	cout << "Loading data is done.\n";
	model->createMPIDist();
	cout << "done12\n";
	model->initWeightAndBcast();
	cout << "done13\n";
	model->sendAndRecvForManager();
	cout << "CPU number: " << omp_get_num_procs() << endl;  
}

void classifyNode(TrainClassification<float> *model){

	cout << "Initialize layers...\n";

	model->createLayerForWorker();
	cout << "Initialize layers is done.\n";
	model->createWBiasForWorker();
	cout << "done2\n";
	model->createPixelAndLabel();
	cout << "done3\n";
	model->createYDEDYForWorker();
	cout << "done4\n";
	model->createMPIDist();
	cout << "done5\n";
	model->initWeightAndBcast();
	cout << "done6\n";
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

	int num_network = num_process/2;
	TrainClassification<float> *DIC_model[num_network];

	string network_json[] = {"script/DIC_seg_32.json", "script/DIC_seg_64.json", \
					"script/DIC_seg_96.json", "script/DIC_seg_160.json", \
					"script/DIC_seg_320.json"};
	string train_file[] = {"../data/DIC_seg_train_32.bin", "../data/DIC_seg_train_64.bin", \
					"../data/DIC_seg_train_96.json", "script/DIC_seg_train_160.json", \
					"../data/DIC_seg_train_320.json"};
	string valid_file[] = {"../data/DIC_seg_valid_32.bin", "../data/DIC_seg_valid_64.bin", \
					"../data/DIC_seg_valid_96.json", "script/DIC_seg_valid_160.json", \
					"../data/DIC_seg_valid_320.json"};

	DIC_model[pid/2] = new TrainClassification<float>((pid/2)*2, pid);
	cout << network_json << endl;
	DIC_model[pid/2]->parseNetJson(network_json[pid/2]);
	DIC_model[pid/2]->parseImgBinary(num_process, train_file[pid/2], valid_file[pid/2]);

	if(pid % 2 == 0){ 
		managerNode(DIC_model[pid/2]);
	}   
	else{
		classifyNode(DIC_model[pid/2]);
	}
	 	
	delete DIC_model[pid/2];

	MPI_Finalize();


	return 0;
}

















