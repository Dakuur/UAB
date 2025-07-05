import random
from time import time
import math

def montecarlo(n):
  inside = 0
  for i in range(n):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    if x**2 + y**2 <= 1:
      inside += 1
  return (inside / n) * 4

def geometry(n):
  a = 1.0
  b = 1.0 / (2 ** 0.5)
  t = 1.0 / 4.0
  p = 1.0

  for i in range(n):
    an = (a + b) / 2
    bn = (a * b) ** 0.5
    tn = t - p * (a - an) ** 2
    pn = 2 * p

    a, b, t, p = an, bn, tn, pn

  return (a + b) ** 2 / (4 * t)

def mostrar_menu():
    print("Quin mètode vols?\n1 - Montecarlo\n2 - Polígons")

mostrar_menu()
func = int(input())
n = int(input("Número de passos: "))

inici = time()
if func == 1:
    res = montecarlo(n)
else:
    res = geometry(n)
final = time()

print(res)
print("N: " + str(n))
print(str(final-inici) + " segons")
print("Error: " + str(abs(math.pi-res)))