from math import gcd

def son_coprimos(a, b):
    return gcd(a, b) == 1

# Ejemplo de uso
num1 = int(input("Introduce el primer número: "))
num2 = int(input("Introduce el segundo número: "))

if son_coprimos(num1, num2):
    print(f"{num1} y {num2} son coprimos.")
else:
    print(f"{num1} y {num2} NO son coprimos.")
