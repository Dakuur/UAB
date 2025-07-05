suma = 0
producte = 1
mitjana = 0

num1 = int(input())
num2 = int(input())
n = num1

divisor = 0

if num1%2 == 0:
    inici = num1
else:
    inici = num1 + 1

if num2%2 == 0:
    final = num2
else:
    final = num2 - 1

#SUMA
for n in range(inici, final+1,2):
    suma = suma + n
    divisor = divisor + 1

#PRODUCTE
for n in range(inici, final+1,2):
    producte = producte*n

#MITJANA
mitjana = suma/divisor

print("Suma:", suma, "- Producte:", producte, "- Mitjana:", mitjana)