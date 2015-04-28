#!/bin/bash
GCC = gcc
CC = g++ -std=c++0x                                                                    
NVCC = nvcc
MPICXX = mpicxx
CCFLAGS = -c
NVCCFLAGS = -g -O3 -c -Xcompiler -fopenmp

LIB = -lcuda -lcudart -lcublas -lm  -lpthread -lmpich -lgomp -lmpi
INCLUDES = -I./include

BUILD_DIR = ./bin
OBJ_DIR = ./obj
SRCS_DIR = ./src
INCLUDES_DIR = ./include
SRCS_TARGET_DIR = ./main_src

CU_INCLUDES = $(shell find $(INCLUDES_DIR) -name "*.cuh")
CXX_INCLUDES = $(shell find $(INCLUDES_DIR) -name "*.h")
HXX_INCLUDES = $(shell find $(INCLUDES_DIR) -name "*.hpp")

HXX_SRCS = $(subst $(INCLUDES_DIR), $(SRCS_DIR), ${HXX_INCLUDES:.hpp=.cpp})
CU_SRCS = $(shell find $(SRCS_DIR) -name "*.cu")
CXX_SRCS = $(filter-out $(HXX_SRCS), $(shell find $(SRCS_DIR) -name "*.cpp"))

CXX_OBJS += $(subst $(SRCS_DIR), $(OBJ_DIR), ${CXX_SRCS:.cpp=.o})
CU_OBJS += $(subst $(SRCS_DIR), $(OBJ_DIR), ${CU_SRCS:.cu=.o})

TARGET ?= main
MULTI_PROCESS ?= 0
MULTI_MECHINE ?= 0
OPEN_MPI ?= 0
NUM_PROCESS ?= 2
BUILD_TARGET = $(BUILD_DIR)/$(TARGET)
SRCS_TARGET = $(SRCS_TARGET_DIR)/$(TARGET).cu
OBJ_TARGET = $(OBJ_DIR)/$(TARGET).o

$(OBJ_DIR)/%.o: $(SRCS_DIR)/%.cpp
	$(CC) $(CCFLAGS) $^ $(INCLUDES) -o $@
	
$(OBJ_DIR)/%.o: $(SRCS_DIR)/%.cu
	$(NVCC) $(CCFLAGS) $^ $(INCLUDES) -o $@

$(BUILD_TARGET): $(CXX_OBJS) $(CU_OBJS) $(SRCS_TARGET)
	$(NVCC) $(NVCCFLAGS) $(SRCS_TARGET) $(INCLUDES) -o $(OBJ_TARGET)
	$(NVCC) -o $(BUILD_TARGET) $(OBJ_TARGET)  $(CXX_OBJS) $(CU_OBJS) $(LIB)  
	
	if [ $(MULTI_MECHINE) -eq 1 ] ; then \
		scp $(BUILD_TARGET) crd@gpu-server1:/home/crd/conv/bin; \
	fi
	if [ $(MULTI_PROCESS) -eq 1 ]; then \
		mpirun_rsh -np $(NUM_PROCESS) -hostfile hostfile MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 $(BUILD_TARGET); \
	fi

			#mpirun -np $(NUM_PROCESS) -hostfile hostfile $(BUILD_TARGET); \
#   scp testMultiGpu root@gpu-server1:/home/conv/src2
#	mpirun_rsh -np 3 -hostfile hostfile MV2_CPU_BINDING_POLICY=scatter valgrind --leak-check=full ./testMultiGpu
#mpirun_rsh -np 2 -hostfile hostfile MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 MV2_USE_GPUDIRECT=1 ./testMultiGpu

cleanall:
	rm -rf $(OBJ_DIR)/*.o  
