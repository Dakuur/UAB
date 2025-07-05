#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank;
    int number;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) { // Procés 0, envía
        number = 12345;
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Soc el procés 0 i he enviat %d.\n", number);
    } else if (rank == 1) { // Procés 1, rep
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Soc el procés 1 i he rebut %d.\n", number);
    }

    MPI_Finalize();
    return 0;
}
