"""import copy as cp 

l1 = ["Hola", "si", "depende"]
l2 = l1
l3 = cp.copy(l1)
l4 = cp.deepcopy(l1)

l2[0] = "Miau"

print(f"l1: {l1}")
print(f"l2: {l2}")
print(f"l3: {l3}")
print(f"l4: {l4}")

a=[1,2]
b=[3,4]
c=[5,6]
l1=[a,b]
l2=l1
l3=list(l1)
print(l1)
print(l2)
print(l3)
l1.append(c)
print(l1)
print(l2)
print(l3)

import copy as cp
b=["HOLA","CASA","AVIO"]
c=[1,2,3]
d=["GROC","BLAU","VERD"]
l2=[b,c,d]
l3=cp.copy(l2)
l4=l2
l5=cp.deepcopy(l2)
l2[1][2]=77
l2.insert(0,2)
def f(v):
 v[0]=9

f(b)
print(b)"""

def factorial(n):
    if n > 1:
        return n*factorial(n-1)
    else:
        return n
    
#print(factorial(435))

from math import log2
from time import time
from copy import copy

def funcIteratiu( n): # O(n)
    cacapol = n/(log2(n))
    teoria = n
    iter = 0
    whiles = 0
    whiles_t = log2(n)

    while (n >= 1): # O(log2(n))

        #print(f"Iteració: {n//2}")
        for i in range(0, n//2): # O(n/(log2(n)))
            iter += 1
        n = n//2
        whiles += 1

    print(f"\nIteracions teoriques: {teoria}")
    print(f"Iteracions reals: {iter}\n")
    print(f"Whiles teorics: {whiles_t}")
    print(f"Whiles reals: {whiles}\n")

    print(f"Pol: {cacapol}")
    print(f"Pol real: {iter/whiles}\n")
"""
start = time()
funcIteratiu(123456)
print(f"Time: {time() - start}")"""

#complexitat = N

"""n = 3465

value = 0
for i in range(n):
  for j in range(i):
    value+=1

print(value)
print((n*(n+1))/2)"""


for i in range(1):
    print("si")