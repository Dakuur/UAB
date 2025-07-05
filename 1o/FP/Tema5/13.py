def Fibonacci(n):
    an = 0
    a0= 0
    a1 = 1
    for i in range(1,n):
        an = a0 + a1
        a0 = a1
        a1 = an
    if n == 1:
        an = 1
    return an

n = int(input())
while n < 0:
    print("Error: El nombre no pot ser negatiu")
    n = int(input())

print("El terme",n,"de la sèrie de Fibonacci és",Fibonacci(n))