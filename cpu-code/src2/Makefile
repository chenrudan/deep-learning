#Filename:Makefile
CXX=g++
CX=mpicxx
CC=mpicc
FC=gfortran
DEBUG= -g
CFLAGS= -Wall -c $(DEBUG) 
LFLAGS= -Wall $(DEBUG)
LDFLAGS=-pthread
MODEL=-std=c++11
HPATH=../include
LIB1=../lib/libcblas.a
LIB2=../lib/libatlas.a

loaddata:
	$(CXX) $(CFLAGS) -I $(HPATH) dataset.cpp dataset_CIFAR_10.cpp dataset_stl10.cpp
	#$(CXX) loaddata.cpp -I$(HPATH) -o loaddata `pkg-config --cflags --libs opencv` dataset.o dataset_CIFAR_10.o dataset_stl10.o
	$(CXX) loaddata.cpp -I$(HPATH) -o loaddata dataset.o dataset_CIFAR_10.o dataset_stl10.o
	
hpc_first:
	$(CX) $(CFLAGS) -I $(HPATH) HPC.cpp train_hpc_firstlayer.cpp 
	$(CX) -o train_hpc_firstlayer HPC.o train_hpc_firstlayer.o $(LIB1) $(LIB2) -lm
	scp train_hpc_firstlayer root@gpu-server3:/home/common/nej/cpluscode/src1	
	mpirun_rsh -np 44 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./train_hpc_firstlayer

hpc_second:
	$(CX) $(CFLAGS) -I $(HPATH) HPC.cpp train_hpc_secondlayer.cpp 
	$(CX) -o train_hpc_secondlayer HPC.o train_hpc_secondlayer.o $(LIB1) $(LIB2) -lm
	scp train_hpc_secondlayer root@gpu-server3:/home/common/nej/cpluscode/src1	
	mpirun_rsh -np 40 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./train_hpc_secondlayer

hpc_third:
	$(CX) $(CFLAGS) -I $(HPATH) HPC.cpp train_hpc_thirdlayer.cpp 
	$(CX) -o train_hpc_thirdlayer HPC.o train_hpc_thirdlayer.o $(LIB1) $(LIB2) -lm
	scp train_hpc_thirdlayer root@gpu-server3:/home/common/nej/cpluscode/src1	
	mpirun_rsh -np 36 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./train_hpc_thirdlayer

hpc_test:
	$(CX) $(CFLAGS) -I $(HPATH) HPC.cpp test_hpc1.cpp 
	$(CX) -o test_hpc1 HPC.o test_hpc1.o $(LIB1) $(LIB2) -lm
#	$(CX) -o test_hpc2 HPC.o test_hpc2.o $(LIB1) $(LIB2) -lm
#	$(CX) -o test_hpc3 HPC.o test_hpc3.o $(LIB1) $(LIB2) -lm
	scp test_hpc1 root@gpu-server3:/home/common/nej/cpluscode/src1	
	mpirun_rsh -np 44 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./test_hpc1
	#mpirun_rsh -np 30 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./test_hpc2
	#mpirun_rsh -np 21 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./test_hpc3

hpc_face1:
	$(CX) $(CFLAGS) -I $(HPATH) HPC.cpp train_face1.cpp 
	$(CX) -o train_face1 HPC.o train_face1.o $(LIB1) $(LIB2) -lm
#	scp train_face1 root@gpu-server2:/home/common/nej/cpluscode/src2
	mpirun_rsh -np 22 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./train_face1

hpc_face2:
	$(CX) $(CFLAGS) -I $(HPATH) HPC.cpp train_face2.cpp 
	$(CX) -o train_face2 HPC.o train_face2.o $(LIB1) $(LIB2) -lm
#	scp train_face2 root@gpu-server3:/home/common/nej/cpluscode/src2
	mpirun_rsh -np 10 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./train_face2

hpc_face3:
	$(CX) $(CFLAGS) -I $(HPATH) HPC.cpp train_face3.cpp 
	$(CX) -o train_face3 HPC.o train_face3.o $(LIB1) $(LIB2) -lm
	scp train_face1 root@gpu-server3:/home/common/nej/cpluscode/src2
	mpirun_rsh -np 22 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./train_face3

lgd:
	$(CXX) $(CFLAGS) -I $(HPATH) LGD.cpp train_logistic.cpp 
	$(CXX) -o train_logistic LGD.o train_logistic.o $(LIB1) $(LIB2) -lm
	./train_logistic

test:
	$(CXX) $(LFLAGS) test.cpp -o test
	./test

cleanall:
	rm -f *.o train_hpc_firstlayer train_hpc_secondlayer train_hpc_thirdlayer test_hpc1 test_hpc2 test_hpc3 test train_logistic train_face1 train_face2 train_face3
