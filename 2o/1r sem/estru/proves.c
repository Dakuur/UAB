#include <stdio.h>

int main() {
    int n; // You need to provide the value of n

    // Initialize the sum
    int sum = 0;

    // Outer loop (i)
    for (int i = 1; i < n; i *= 2) {
        // Middle loop (j)
        for (int j = n; j > 0; j /= 2) {
            // Inner loop (k)
            for (int k = j; k < n; k += 2) {
                sum += (i + j * k);
            }
        }
    }

    // Print the final sum
    printf("Sum: %d\n", sum);

    return 0;
}
