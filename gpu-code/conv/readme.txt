

#安装&运行环境

	1.cuda-6.5   
		链接 -lcuda -lcudart -lcublas
	2.NVIDIA驱动（安装cuda时默认版本即可）
	3.mvapich2.2
		链接 -lmpich
	4.Mellanox OFED适配器
	5.vargrind（非必要，查看内存泄漏）
		运行 valgrind --leak-check=full
	6.pthread是由linux提供的POSIX线程标准的库
		头文件 pthread.h
		链接 -lpthread
	7.openmp共享内存方式的多线程并发的编程API
		编译 -fopenmp
		链接 -lgomp
	8.运行的时候加上MV2_ENABLE_AFFINITY=0，使MPI能够实现多线程控制
	9.MV2_USE_CUDA=1，mpi能够调用kernel函数

#运行
	#testLogistic,testConv是单gpu版本
	#testMultipu，是多gpu的logsitic
	#main，是多机多gpu的卷积，目前仍在调试，可能无法使用
	make xxx
	mpirun_rsh -np 2 -hostfile hostfile MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 ./xxx
	

#注意事项
	
基本数据类型Matrix，NVmatix，类似矩阵行列存储，行主序
	input存储，minibatchSize * (28 * 28)
	w存储，numFilters * (5 * 5)
	bias, numFilters * 1
	
在实现kernel函数，使用了宏定义，这个是由于shared memory在分配的时候要固定大小，这个以后需要改正

在执行程序时
	make TARGET=main MULTI_PROCESS=1 MULTI_MECHINE=1
