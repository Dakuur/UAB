cadena = str(input())

posicio = int(input())

while posicio in range(0,len(cadena)):
    caracter = str(input())
    abans = cadena[:posicio]
    despres = cadena[posicio+1:]
    cadena = abans + caracter + despres
    posicio = int(input())

print(cadena)