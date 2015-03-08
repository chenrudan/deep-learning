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
	mpirun_rsh -np 3 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./testMultiGpu

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

testGpuSuccess1: $(OBJS) testGpuSuccess1.cu
	$(NVCC) $(NVCCFLAGS) testGpuSuccess1.cu $(INCLUDES)
	$(NVCC) -o testGpuSuccess1 testGpuSuccess1.o $(OBJS) $(LIB)
#   scp testMultiGpu root@gpu-server1:/home/conv/src2
	mpirun_rsh -np 3 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./testGpuSuccess1


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
