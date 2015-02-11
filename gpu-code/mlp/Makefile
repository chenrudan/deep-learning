GCC = gcc
CC = g++    
NVCC = nvcc
LIB = -lcublas

main: main.cu
	$(NVCC) -g -o main main.cu mlp.cu mlp_kernel.cu $(LIB)

mlp: mlp.c
	$(GCC) -g -o mlp mlp.c -lm 

cleanall:
	rm main mlp
