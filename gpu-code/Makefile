CC = g++                                                                                                                                                                            
NVCC = nvcc
LIB = -lcublas


testsgemm: testsgemm.cu
	$(NVCC) -g -o testsgemm testsgemm.cu $(LIB)

cleanall:
	rm testsgemm
