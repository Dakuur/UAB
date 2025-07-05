def main():
    n = int(input("Enter the value of n: "))  # You need to provide the value of n

    # Initialize the sum
    sum_value = 0
    total = 0
    i = j = n
    # Outer loop (i)
    for i in range(1, n, i * 2):
        # Middle loop (j)
        for j in range(n, 0, j // 2):
            # Inner loop (k)
            for k in range(j, n, 2):
                total += 1

    # Print the final sum
    print(f"Sum: {sum_value}")
    print(f"Iteracions: {total}")

if __name__ == "__main__":
    main()
