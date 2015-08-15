///
/// \file mpi_distribute.hpp
/// \brief 继承数据类，拥有矩阵的特性
///


#ifndef MPIDISTRIBUTE_H_
#define MPIDISTRIBUTE_H_

#include "mpi.h"
#include "model_component.hpp"

#define PROCESS_END 100000;

using namespace std;

/// \brief 实现了数据分配服务器，包括输入像素数据的分派和网络权重数据的交换
///
template<typename Dtype>
class MPIDistribute {
private:
    Dtype *_data; ///>存放数据的指针，它只是一个指针保存了需要传递数据的地址
    int _len;
    int _tag;  ///>代表它是哪个数据，从而可以计算在传递过程中的tag
    int _flag; ///>表示是继续传递数据还是停止，这个由执行进程传递给控制进程
    int _pid; ///>跟哪个进程在交互数据
    MPI_Status _status;
    MPI_Datatype mpi_type;
    int _position;

public:
    MPIDistribute(const int len, const int tag, const int pid, const MPI_Datatype mpi_type, \
        Dtype *server_data = NULL ) : \
 		_len(len), _tag(tag), _pid(pid), _server_data(server_data), \
		_mpi_type(mpi_type) {}
    ~MPIDistribute() {}

    void receviceFlag();
    void sendFlag();
    void dataTo();
    void dataFrom();
    ///>初始化时需要pack的数据MPI类型就是MPI_PACKED
    void packAndSend(const int num, const int *len, Dtype **data);
    void recvAndUnpack(const int num, const int *len, Dtype **data);
};

#include "../src/mpi_distribute.cpp"

#endif