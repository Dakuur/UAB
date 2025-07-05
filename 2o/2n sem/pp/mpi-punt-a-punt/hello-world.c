#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]){
    
    int rank, size;

    if (MPI_Init(&argc, &argv) != MPI_SUCCESS){
        printf("Error inicialitzant. Abortant!\n"); exit(1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Hello world from %d of %d!!\n", rank, size);

    MPI_Finalize();
}