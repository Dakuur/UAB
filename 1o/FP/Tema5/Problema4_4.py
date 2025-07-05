from time import time

def factorial(num):
    resultat = 1
    for i in range(num, 0, -1):
        resultat = resultat * i
    return resultat

def sumatori(num):
    resultat = 0
    for i in range(num, 0, -1):
        resultat = resultat + i
    return resultat

def Timer(fnc,arg):
    t0=time()
    fnc(arg)
    t1=time()
    return t1-t0

print(Timer(sumatori, 10000))