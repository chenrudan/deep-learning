GCC = gcc
CC = g++                                                                        
NVCC = nvcc
MPICXX = mpicxx
NVCCFLAGS= -g -O3 -c 

LDFLAGS = -L/usr/lib/openmpi/lib
LIB = -lcuda -lcudart -lcublas -lm  -lpthread -lmpi
INCLUDES = -I../include -I/usr/lib/openmpi/include 

testMultiGpu:
	$(NVCC) $(NVCCFLAGS) testMultiGpu.cu matrix.cpp nvmatrix.cu nvmatrix_kernel.cu convnet.cu convnet_kernel.cu $(INCLUDES)
	$(NVCC) -o testMultiGpu *.o  $(LDFLAGS) $(LIB)
#   scp train_ocean1 root@gpu-server2:/home/common/nej/cpluscode/src2
#	mpirun -np 2 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter ./testMultiGpu


main: main.cu
	$(NVCC) -g -o main main.cu matrix.cpp nvmatrix.cu nvmatrix_kernel.cu convnet.cu convnet_kernel.cu $(LDFLAGS)

test: test.cu
	$(NVCC) -g -o test test.cu matrix.cpp nvmatrix.cu nvmatrix_kernel.cu $(LDFLAGS)


testLogistic: testLogistic.cu
	$(NVCC) -g -o  testLogistic testLogistic.cu matrix.cpp nvmatrix.cu nvmatrix_kernel.cu convnet.cu convnet_kernel.cu $(INCLUDES) $(LDFLAGS)

testpy:
	THEANO_FLAGS='blas.ldflags=' python logistic_sgd.py 

cleanall:
	rm -rf *.o main test nohup.out testLogistic testMultiGpu
