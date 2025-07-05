#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

float stencil(float v1, float v2, float v3, float v4) {
    return (v1 + v2 + v3 + v4) / 4;
}

void laplace_step(float *in, float *out, int start_row, int end_row, int m) {
    int i, j;
    for (i = start_row; i < end_row; i++)
        for (j = 1; j < m - 1; j++)
            out[i * m + j] = stencil(in[i * m + j + 1], in[i * m + j - 1], in[(i - 1) * m + j], in[(i + 1) * m + j]);
}

float laplace_error(float *old, float *new, int start_row, int end_row, int m) {
    int i, j;
    float error = 0.0f;
    for (i = start_row; i < end_row; i++)
        for (j = 1; j < m - 1; j++)
            error = fmaxf(error, sqrtf(fabsf(old[i * m + j] - new[i * m + j])));
    return error;
}

void laplace_copy(float *in, float *out, int start_row, int end_row, int m) {
    int i, j;
    for (i = start_row; i < end_row; i++)
        for (j = 1; j < m - 1; j++)
            out[i * m + j] = in[i * m + j];
}

void laplace_init(float *in, int n, int m) {
    int i, j;
    const float pi = 2.0f * asinf(1.0f);
    memset(in, 0, n * m * sizeof(float));
    for (j = 0; j < m; j++) in[j] = 0.f;
    for (j = 0; j < m; j++) in[(n - 1) * m + j] = 0.f;
    for (i = 0; i < n; i++) in[i * m] = sinf(pi * i / (n - 1));
    for (i = 0; i < n; i++) in[i * m + m - 1] = sinf(pi * i / (n - 1)) * expf(-pi);
}

int main(int argc, char** argv) {
    int n = 4096, m = 4096;
    const float pi = 2.0f * asinf(1.0f);
    const float tol = 3.0e-3f;
    int world_size, rank;

    float error = 1.0f;

    int i, j, iter_max = 100, iter = 0;
    float *A, *Anew;

    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Start timer
    start_time = MPI_Wtime();

    // get runtime arguments: n, m and iter_max
    if (argc > 1) { n = atoi(argv[1]); }
    if (argc > 2) { m = atoi(argv[2]); }
    if (argc > 3) { iter_max = atoi(argv[3]); }

    int rows_per_node = n / world_size;
    int start_row = rank * rows_per_node + 1;
    int end_row = (rank + 1) * rows_per_node;
    float local_error;

    if (rank == world_size - 1) end_row = n - 1;

    A = (float*)malloc(n * m * sizeof(float));
    Anew = (float*)malloc(n * m * sizeof(float));

    // set boundary conditions
    laplace_init(A, n, m);

    if (rank == 0) {
        printf("Parallel code results with %d nodes:\n\n", world_size);
        printf("Jacobi relaxation Calculation: %d rows x %d columns mesh, maximum of %d iterations\n",
               n, m, iter_max);
    }

    // Main loop: iterate until error <= tol a maximum of iter_max iterations
    while (error > tol && iter < iter_max) {
        // Send top and bottom rows with neighboring processes
        if (rank > 0) {
            MPI_Send(&A[start_row * m], m, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&A[(start_row - 1) * m], m, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < world_size - 1) {
            MPI_Send(&A[(end_row - 1) * m], m, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&A[end_row * m], m, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Compute new values using main matrix and writing into auxiliary matrix
        laplace_step(A, Anew, start_row, end_row, m);

        // Compute error = maximum of the square root of the absolute differences
        local_error = laplace_error(A, Anew, start_row, end_row, m);

        // Copy from auxiliary matrix to main matrix
        laplace_copy(Anew, A, start_row, end_row, m);

        // Reduce the error of all nodes
        MPI_Allreduce(&local_error, &error, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

        // if number of iterations is multiple of 10 then print error on the screen
        iter++;
        if (rank == 0 && iter % (iter_max / 10) == 0)
            printf("%5d, %0.6f\n", iter, error);
    } // while

    // End timer
    end_time = MPI_Wtime();

    // Print the elapsed time
    if (rank == 0) {
        printf("Total computation time: %f seconds\n", end_time - start_time);
    }

    free(A);
    free(Anew);

    MPI_Finalize();
    return 0;
}