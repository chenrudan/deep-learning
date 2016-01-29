#!/bin/bash
GCC = gcc
CC = g++ -std=c++0x                                                                    
NVCC = nvcc
CCFLAGS = -c -pg
NVCCFLAGS = -g -pg -O3 -c
PTXFLAGES = --machine 64

LIB = -L/usr/local/cuda/lib64 -lcuda -lcudart  -lcublas -lm
INCLUDES = -I./include

BUILD_DIR = ./bin
OBJ_DIR = ./obj
SRCS_DIR = ./src
INCLUDES_DIR = ./include
SRCS_TARGET_DIR = ./main_src

CU_INCLUDES = $(shell find $(INCLUDES_DIR) -name "*.cuh")
CXX_INCLUDES = $(shell find $(INCLUDES_DIR) -name "*.h")
HXX_INCLUDES = $(shell find $(INCLUDES_DIR) -name "*.hpp")

#HXX对应的cpp,cu，然后将它们排除掉，不进行编译
HXX_SRCS = $(subst $(INCLUDES_DIR), $(SRCS_DIR), ${HXX_INCLUDES:.hpp=.cpp})
HXX_SRCS += $(subst $(INCLUDES_DIR), $(SRCS_DIR), ${HXX_INCLUDES:.hpp=.cu})
CU_SRCS = $(filter-out $(HXX_SRCS), $(shell find $(SRCS_DIR) -name "*.cu"))
CXX_SRCS = $(filter-out $(HXX_SRCS), $(shell find $(SRCS_DIR) -name "*.cpp"))
CU_HPP_SRCS = $(filter $(HXX_SRCS), $(shell find $(SRCS_DIR) -name "*.cu"))
CXX_HPP_SRCS = $(filter $(HXX_SRCS), $(shell find $(SRCS_DIR) -name "*.cpp"))

#生成的链接文件
CXX_OBJS += $(subst $(SRCS_DIR), $(OBJ_DIR), ${CXX_SRCS:.cpp=.o})
CU_OBJS += $(subst $(SRCS_DIR), $(OBJ_DIR), ${CU_SRCS:.cu=.o})

TARGET ?= main
MULTI_PROCESS ?= 1
MULTI_MECHINE ?= 0
OPEN_MPI ?= 0
NUM_PROCESS ?= 2
BUILD_TARGET = $(BUILD_DIR)/$(TARGET)
SRCS_TARGET = $(SRCS_TARGET_DIR)/$(TARGET).cu
OBJ_TARGET = $(OBJ_DIR)/$(TARGET).o

#print: $(CXX_SRCS) $(CU_SRCS) 
#	echo $(HXX_SRCS)
#	echo $(CXX_SRCS)
#	echo $(CU_SRCS)

$(OBJ_DIR)/%.o: $(SRCS_DIR)/%.cpp
	$(CC) $(CCFLAGS) $^ $(INCLUDES) -o $@ 
	
$(OBJ_DIR)/%.o: $(SRCS_DIR)/%.cu
	$(NVCC) $(CCFLAGS) $^ $(INCLUDES) -o $@

$(BUILD_TARGET): $(CXX_OBJS) $(CU_OBJS) $(SRCS_TARGET) $(CU_HPP_SRCS) $(CXX_HPP_SRCS)
	$(NVCC) $(NVCCFLAGS) $(SRCS_TARGET) $(INCLUDES) -o $(OBJ_TARGET)
	$(NVCC) -o $(BUILD_TARGET) $(OBJ_TARGET) $(CXX_OBJS) $(CU_OBJS) $(LIB) $(INCLUDES)
	
cleanall:
	rm -rf $(OBJ_DIR)/*.o  
