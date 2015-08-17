///
/// \file mpi_distribute.cpp
/// @brief

#include "mpi_distribute.hpp"

using namespace std;

template <typename Dtype>
void MPIDistribute<Dtype>::receviceFlag(){
	MPI_Recv(&_flag, 1, MPI_INT, _pid, _tag, MPI_COMM_WORLD, &_status);
}

template <typename Dtype>
void MPIDistribute<Dtype>::sendFlag(int flag){
	_flag = flag;
	MPI_Send(&_flag, 1, MPI_INT, _pid, _tag, MPI_COMM_WORLD);
}

template <typename Dtype>
void MPIDistribute<Dtype>::dataTo(){
	MPI_Send(_data, _len, _mpi_type, _pid, _flag+_tag, MPI_COMM_WORLD);
}

template <typename Dtype>
void MPIDistribute<Dtype>::dataFrom(){
	MPI_Recv(_data, _len, _mpi_type, _pid, _flag+_tag, MPI_COMM_WORLD, &_status);
}

template <typename Dtype>
void MPIDistribute<Dtype>::bcast(){
	MPI_Bcast(_data, _len, _mpi_type, _pid, MPI_COMM_WORLD);
}

template <typename Dtype>
void MPIDistribute<Dtype>::packAndSend(const int num, const int *len, Dtype **data, \
			const MPI_Datatype mpi_type){
	int buff_len = 0;
	for (int i = 0; i < num; ++i){
		buff_len += len[i]*sizeof(mpi_type);
	}
	_position = 0;
	char buff[buff_len];
	for (int i = 0; i < num; ++i) {
		MPI_Pack(data[i], len[i], mpi_type, buff, buff_len, &_position, \
                  MPI_COMM_WORLD);
	}
	_data = buff;
	_len = _position;
	dataTo();
}

template <typename Dtype>
void MPIDistribute<Dtype>::recvAndUnpack(const int num, const int *len, \
			Dtype **data, const MPI_Datatype mpi_type){
	int buff_len = 0;
	for (int i = 0; i < num; ++i){
		buff_len += len[i]*sizeof(mpi_type);
	}
	_position = 0;
	char buff[buff_len];
	_data = buff;
	_position = 0;

	dataFrom();
	for (int i = 0; i < num; ++i) {
		MPI_Unpack(buff, buff_len, &_position, data[i], len[i], \
				mpi_type, MPI_COMM_WORLD);
	}
}

















