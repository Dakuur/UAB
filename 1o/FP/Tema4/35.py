ordenada = True
num1 = int(input())

for i in range(0,9):
    num2 = int(input())
    if num2 < num1:
        ordenada = False
    num1 = num2

if ordenada == True:
    print("La llista està ordenada.")
else:
    print("La llista no està ordenada.")