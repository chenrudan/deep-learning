GCC = gcc
CC = g++                                                                        
NVCC = nvcc
LIB = -lcublas -lm  

main: main.cu
	$(NVCC) -g -o main main.cu matrix.cpp nvmatrix.cu nvmatrix_kernel.cu convnet.cu convnet_kernel.cu $(LIB)

test: test.cu
	$(NVCC) -g -o test test.cu matrix.cpp nvmatrix.cu nvmatrix_kernel.cu $(LIB)


testLogistic: testLogistic.cu
	$(NVCC) -g -o testLogistic testLogistic.cu matrix.cpp nvmatrix.cu nvmatrix_kernel.cu convnet.cu convnet_kernel.cu $(LIB)

testpy:
	THEANO_FLAGS='blas.ldflags=' python logistic_sgd.py 

cleanall:
	rm -rf main test nohup.out testLogistic
