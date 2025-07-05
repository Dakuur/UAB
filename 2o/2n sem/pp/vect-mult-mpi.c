#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SIZE 1000000

float V1[SIZE], V2[SIZE];

void Init(float V1[], float V2[]) {
    for (unsigned i = 0; i < SIZE; i++) {
        V1[i] = i * 1.1;
        V2[i] = ((i % 2) ? -1.0 : 0.99999999) * i;
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    float local_res = 0.0, res = 0.0;

    //------------------ PRINCIPI SECCIO MULTI NODE -----------------//

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        Init(V1, V2);
    }

    int local_size = SIZE / size;
    float* local_V1 = malloc(local_size * sizeof(float));
    float* local_V2 = malloc(local_size * sizeof(float));

    MPI_Scatter(V1, local_size, MPI_FLOAT, local_V1, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(V2, local_size, MPI_FLOAT, local_V2, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_size; i++) {
        local_res += local_V1[i] * local_V2[i];
    }

    MPI_Reduce(&local_res, &res, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Resultat: %e\n", res);
    }

    free(local_V1);
    free(local_V2);

    MPI_Finalize();

    //-------------------- FI SECCIO MULTI NODE -------------------//

    return 0;
}
