num = int(input())

sumatori = 0

while num <= 0:
    print("Error: el valor ha de ser més gran que zero")
    num = int(input())

inicial = num

for num in range(0, inicial+1):
    sumatori = sumatori + num
    num = num - 1

print("El sumatori de", str(inicial), "és:", str(sumatori))