
total = 0

def f(n):
    global total

    if n == 0:
        return 1
    total += 1
    sum = 0
    for i in range(n):
        sum += f(n-1)
    return sum

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

n = 7
print(f"N: {n}")

f(n)

print(f"Recursions: {total}")

print(f"{n} factorial: {factorial(n)}")