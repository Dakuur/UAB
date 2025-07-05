palindrom = True

paraula = str(input())

for i in range(0, int(len(paraula)/2)):
    if paraula[i] != paraula[len(paraula) - i -1]:
        palindrom = False
        break

if palindrom == True:
    print("La cadena és palíndrom.")
else:
    print("La cadena NO és palíndrom.")