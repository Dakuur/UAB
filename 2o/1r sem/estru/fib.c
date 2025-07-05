#include <stdio.h>
#include <stdlib.h>
#include <time.h>

long limit, i = 0;

// Function to perform Fibonacci calculation
void fibonacci(long n, long *result) {
    long a = 1, b = 0, c;

    for (i = 0; i < n; i++) {
        // Perform the Fibonacci Calculation
        c = a + b;
        a = b;
        b = c;
    }

    *result = b;
}

int main(int argc, char *argv[]) {
    // Get User Input
    if (argc != 2) {
        printf("Improper input. Exiting.\n");
        return -1;
    }

    limit = strtol(argv[1], NULL, 10);

    // Start timing
    clock_t start_time = clock();

    long result;
    fibonacci(limit, &result);

    // End timing
    clock_t end_time = clock();

    // Print the results to stdout
    printf("Fibonacci Number %ld: %ld\n", i, result);

    // Print time taken
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Calculation Time: %f seconds\n", time_taken);

    return 0;
}
