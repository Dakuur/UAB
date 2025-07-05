llista = []

for i in range(0,10):
    llista.append(float(input()))

suma = 0

posicio = 0

while suma <= 25 and posicio < 10:
    suma = suma + llista[posicio]
    posicio = posicio + 1

if posicio >= 10 and suma <= 25:
    print("La suma acumulada de la llista és inferior o igual a 25")
else:
    print("A la posició", str(posicio), "la suma acumulada és superior a 25")