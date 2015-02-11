/*filename: testmpi.cpp*/
#include<iostream>
#include"mpi.h"
using namespace std;

int main()
{
    int nnode, me;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nnode);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    int size = 10000;
    float *a = new float[size];
    float *b = new float[size];
    for(int i = 0; i < size; i++)
    {
        a[i] = i;
    }
    MPI_Allreduce(a, b, size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}








