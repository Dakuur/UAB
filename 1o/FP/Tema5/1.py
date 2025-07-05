def factorial(n):
    fact = 1
    for i in range(n, 0, -1):
        fact = fact*n
        n = n - 1
    return fact

n = int(input("Jugadors a la plantilla: "))
m = int(input("Jugadors simultanis: "))

resultat=factorial(n)/(factorial(m)*factorial(n-m))

print("El nombre d'equips que es poden formar és",int(resultat))