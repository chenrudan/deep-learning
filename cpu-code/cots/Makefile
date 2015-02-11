#################################                                                                                                                                                 
#  Compiler, tools and options  #
#################################

CXX = g++ 
DEBUG = -g
CFLAGES = -Wall -c $(DEBUG)
LFLAGES = -Wall $(DEBUG)
LINK = g++ 
CXXFLAGES = -O3 -DADD_
MC = mpicc
MCXX=mpicxx
BLASPATH = /opt/atlas/
INCLUDE = $(BLASPATH)include
CBLIB = $(BLASPATH)lib/libcblas.a $(BLASPATH)lib/libatlas.a
LIB1=./lib/libcblas.a
LIB2=./lib/libatlas.a
####### Compile
all: mnist_layer1 mnist_runlayer1

layer1: layer1.cpp cots.cpp
	$(MCXX) $(CFLAGES) layer1.cpp cots.cpp load.cpp -I $(INCLUDE)
	$(MCXX) -o layer1 layer1.o load.o cots.o $(LIB1) $(LIB2) -lm 

runlayer1:
	#scp layer1 root@gpu-server1:/home/common/crd/cots
	mpirun_rsh -np 22 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./layer1
    
mnist_layer1: mnist_layer1.cpp cots.cpp
	$(MCXX) $(CFLAGES) mnist_layer1.cpp cots.cpp load.cpp -I $(INCLUDE)
	$(MCXX) -o mnist_layer1 mnist_layer1.o load.o cots.o -pg $(LIB1) $(LIB2) -lm 

mnist_runlayer1:
#	scp mnist_layer1 root@gpu-server3:/home/common/crd/cots
	mpirun_rsh -np 10 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./mnist_layer1
#	scp root@gpu-server3:/home/common/crd/cots/binaryfile/weight/mnist_layer1_*.bin ./binaryfile/weight/
	mv ./binaryfile/mnist_layer1_out.bin ./binaryfile/mnist_layer2_in.bin

mnist_layer2: mnist_layer2.cpp cots.cpp
	$(MCXX) $(CFLAGES) mnist_layer2.cpp cots.cpp load.cpp -I $(INCLUDE)
	$(MCXX) -o mnist_layer2 mnist_layer2.o load.o cots.o -pg $(LIB1) $(LIB2) -lm 

mnist_runlayer2:
	#scp layer1 root@gpu-server1:/home/common/crd/cots
	mpirun_rsh -np 6 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./mnist_layer2

layer2: layer2.cpp
	$(MCXX) $(CFLAGES) layer2.cpp cots.cpp load.cpp -I $(INCLUDE)
	$(MCXX) $(CFLAGES) layer2.cpp -I $(INCLUDE)
	$(MCXX) -o layer2 layer2.o load.o cots.o $(LIB1) $(LIB2) -lm 

runlayer2:
	#scp layer2 root@gpu-server1:/home/common/crd/cots
	mpirun_rsh -np 35 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./layer2

layer3: layer3.cpp
	$(MCXX) $(CFLAGES) layer3.cpp cots.cpp load.cpp -I $(INCLUDE)
	$(MCXX) $(CFLAGES) layer3.cpp -I $(INCLUDE)
	$(MCXX) -o layer3 layer3.o load.o cots.o $(LIB1) $(LIB2) -lm 

runlayer3:
	#scp layer2 root@gpu-server1:/home/common/crd/cots
	mpirun_rsh -np 21 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./layer3

face1:
	$(MCXX) $(CFLAGES) face1.cpp cots.cpp load.cpp -I $(INCLUDE)
	$(MCXX) -o face1 face1.o load.o cots.o -pg $(LIB1) $(LIB2) -lm 
	scp face1 root@gpu-server3:/home/common/crd/cots
	mpirun_rsh -np 22 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./face1
	mv ./binaryfile/face1_out.bin ./binaryfile/face2_in.bin



cleanall:
	rm -f *.o layer1 layer2 layer3 mnist_layer1 face1
