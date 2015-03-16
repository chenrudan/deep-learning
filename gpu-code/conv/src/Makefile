GCC = gcc
CC = g++                                                                        
NVCC = nvcc
MPICXX = mpicxx
CCFLAGS = -c
NVCCFLAGS = -g -O3 -c 

LIB = -lcuda -lcudart -lcublas -lm  -lpthread -lmpich
INCLUDES = -I../include

#CUFILES = $(shell find . -name "*.cu")
#CCFILES = $(shell find . -name "*.cpp")

#OBJS += $(patsubst %.cpp, %.o, $(CCFILES))
#OBJS += $(patsubst %.cu, %.o, $(CUFILES))

OBJS = matrix.o nvmatrix.o nvmatrix_kernel.o convnet.o convnet_kernel.o

testMultiGpu: $(OBJS) testMultiGpu.cu
	$(NVCC) $(NVCCFLAGS) testMultiGpu.cu $(INCLUDES)
	$(NVCC) -o testMultiGpu testMultiGpu.o $(OBJS) $(LIB)
#   scp testMultiGpu root@gpu-server1:/home/conv/src2
#	mpirun_rsh -np 3 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter valgrind --leak-check=full ./testMultiGpu
#mpirun_rsh -np 3 -hostfile hostfile MV2_ENABLE_AFFINITY=0 ./testMultiGpu

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

testGpu2Pro: $(OBJS) testGpu2Pro.cu
	$(NVCC) $(NVCCFLAGS) testGpu2Pro.cu $(INCLUDES)
	$(NVCC) -o testGpu2Pro testGpu2Pro.o $(OBJS) $(LIB)
	mpirun_rsh -np 2 -hostfile hostfile MV2_ENABLE_AFFINITY=0 ./testGpu2Pro

testGpuSuccess1: $(OBJS) testGpuSuccess1.cu
	$(NVCC) $(NVCCFLAGS) testGpuSuccess1.cu $(INCLUDES)
	$(NVCC) -o testGpuSuccess1 testGpuSuccess1.o $(OBJS) $(LIB)
#   scp testMultiGpu root@gpu-server1:/home/conv/src2
	mpirun_rsh -np 2 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./testGpuSuccess1

testGpuSuccess2: $(OBJS) testGpuSuccess2.cu
	$(NVCC) $(NVCCFLAGS) testGpuSuccess2.cu $(INCLUDES)
	$(NVCC) -o testGpuSuccess2 testGpuSuccess2.o $(OBJS) $(LIB)
	mpirun_rsh -np 3 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./testGpuSuccess2

testGpuSuccess3: $(OBJS) testGpuSuccess3.cu
	$(NVCC) $(NVCCFLAGS) testGpuSuccess3.cu $(INCLUDES)
	$(NVCC) -o testGpuSuccess3 testGpuSuccess3.o $(OBJS) $(LIB)
	mpirun_rsh -np 3 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./testGpuSuccess3

main: main.cu
	$(NVCC) -g -o main main.cu matrix.cpp nvmatrix.cu nvmatrix_kernel.cu convnet.cu convnet_kernel.cu  $(INCLUDES) $(LIB)

test: test.cu
	$(NVCC) -g -o test test.cu matrix.cpp nvmatrix.cu nvmatrix_kernel.cu


testLogistic: testLogistic.cu
	$(NVCC) -g -o  testLogistic testLogistic.cu matrix.cpp nvmatrix.cu nvmatrix_kernel.cu convnet.cu convnet_kernel.cu $(INCLUDES) $(LIB)

testpy:
	THEANO_FLAGS='blas.ldflags=' python logistic_sgd.py 

cleanall:
	rm -rf *.o main test nohup.out testLogistic testMultiGpu
