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
LIB1=../cots/lib/libcblas.a
LIB2=../cots/lib/libatlas.a
####### Compile
all: testmpi runtest

testmpi:
	$(MCXX) $(CFLAGES) testmpi.cpp -I $(INCLUDE)
	$(MCXX) -o testmpi testmpi.o $(LIB1) $(LIB2) -lm 

runtest:
	scp testmpi root@gpu-server1:/home/common/crd/test
	mpirun_rsh -np 5 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./testmpi
    
cleanall:
	rm -f *.o testmpi
