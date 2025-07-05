def fib(x):
    a, b = 0, 1
    for i in range(x):
        yield a
        a, b = b, a + b

fib_gen = fib(20)

for i in fib_gen:
    print(i)