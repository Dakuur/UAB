llista = []
for i in range(0, 12):
    llista.append(input())
print("Entrada:",' '.join(llista))
for i in range(0, len(llista)):
    if int(llista[i]) < 0:
        llista[i] = int(llista[i])
        llista[i] = 0
#print("Sortida:",' '.join(llista))
print("Sortida:",llista)