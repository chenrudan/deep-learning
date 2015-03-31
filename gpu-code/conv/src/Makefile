GCC = gcc
CC = g++                                                                        
NVCC = nvcc
MPICXX = mpicxx
CCFLAGS = -c
NVCCFLAGS = -g -O3 -c -Xcompiler -fopenmp

LIB = -lcuda -lcudart -lcublas -lm  -lpthread -lmpich -lgomp
INCLUDES = -I../include

#CUFILES = $(shell find . -name "*.cu")
#CCFILES = $(shell find . -name "*.cpp")

#OBJS += $(patsubst %.cpp, %.o, $(CCFILES))
#OBJS += $(patsubst %.cu, %.o, $(CUFILES))

OBJS = matrix.o nvmatrix.o nvmatrix_kernel.o convnet.o convnet_kernel.o logistic.o logistic_kernel.o 

main: $(OBJS) main.cu
	$(NVCC) $(NVCCFLAGS) main.cu $(INCLUDES)
	$(NVCC) -o main main.o $(OBJS) $(LIB)  
	#scp main crd@gpu-server1:/home/crd/conv/src
	mpirun_rsh -np 2 -hostfile hostfile MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 MV2_USE_GPUDIRECT=1 ./main

test: $(OBJS) test.cu
	$(NVCC) $(NVCCFLAGS) test.cu $(INCLUDES)
	$(NVCC) -o test test.o $(OBJS) $(LIB)

testLogistic: testLogistic.cu
	$(NVCC) $(NVCCFLAGS) testLogistic.cu $(INCLUDES)
	$(NVCC) -o testLogistic testLogistic.o $(OBJS) $(LIB)

testConv: $(OBJS) testConv.cu
	$(NVCC) $(NVCCFLAGS) testConv.cu $(INCLUDES)
	$(NVCC) -o testConv testConv.o $(OBJS) $(LIB)

testMultiGpu: $(OBJS) testMultiGpu.cu
	$(NVCC) $(NVCCFLAGS) testMultiGpu.cu $(INCLUDES)
	$(NVCC) -o testMultiGpu testMultiGpu.o $(OBJS) $(LIB)
#   scp testMultiGpu root@gpu-server1:/home/conv/src2
#	mpirun_rsh -np 3 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter valgrind --leak-check=full ./testMultiGpu
#mpirun_rsh -np 2 -hostfile hostfile MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 MV2_USE_GPUDIRECT=1 ./testMultiGpu

matrix.o: matrix.cpp
	$(CC) $(CCFLAGS) $^ $(INCLUDES) 

nvmatrix.o: nvmatrix.cu
	$(NVCC) $(NVCCFLAGS) $^ $(INCLUDES)

nvmatrix_kernel.o: nvmatrix_kernel.cu
	$(NVCC) $(NVCCFLAGS) $^ $(INCLUDES)
	
convnet.o: convnet.cu
	$(NVCC) $(NVCCFLAGS) $^ $(INCLUDES)
	 
convnet_kernel.o: convnet_kernel.cu
	$(NVCC) $(NVCCFLAGS) $^ $(INCLUDES)

logistic.o: logistic.cu
	$(NVCC) $(NVCCFLAGS) $^ $(INCLUDES)
	 
logistic_kernel.o: logistic_kernel.cu
	$(NVCC) $(NVCCFLAGS) $^ $(INCLUDES)


	
testGpuSuccess4: $(OBJS) testGpuSuccess4.cu
	$(NVCC) $(NVCCFLAGS) testGpuSuccess4.cu $(INCLUDES)
	$(NVCC) -o testGpuSuccess4 testGpuSuccess4.o $(OBJS) $(LIB)

testGpuSuccess1: $(OBJS) testGpuSuccess1.cu
	$(NVCC) $(NVCCFLAGS) testGpuSuccess1.cu $(INCLUDES)
	$(NVCC) -o testGpuSuccess1 testGpuSuccess1.o $(OBJS) $(LIB)


testGpuSuccess3: $(OBJS) testGpuSuccess3.cu
	$(NVCC) $(NVCCFLAGS) testGpuSuccess3.cu $(INCLUDES)
	$(NVCC) -o testGpuSuccess3 testGpuSuccess3.o $(OBJS) $(LIB)


testpy:
	THEANO_FLAGS='blas.ldflags=' python logistic_sgd.py 

cleanall:
	rm -rf *.o 
