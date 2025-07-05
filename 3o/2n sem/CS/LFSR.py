seed = 0b10111010011100110000

result = []

for i in range(2000):
    lsb = seed & 1  # Get the least significant bit
    seed = seed >> 1  # Shift right by 1
    if lsb == 1:  # If the lsb was 1
        seed = seed ^ 0b1011  # XOR with the polynomial
    print(f"{i:2d}: {seed & 1}")
    result.append(seed & 1)

print(f"1s: {result.count(1)}")
print(f"0s: {result.count(0)}")
