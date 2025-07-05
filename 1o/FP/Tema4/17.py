num_elements = int(input())

llista = []

for i in range(0, num_elements):
    llista.append(int(input()))

print("1-Afegir element al final la llista\n2-Afegir element en una posició de la llista\n3-Eliminar el darrer element de la llista\n4-Eliminar l'element d'una posició de la llista\n5-Eliminar la primera aparició d'un valor a la llista\n6-Sortir")

opcio = int(input())

while opcio != 6:
    if opcio == 1:
        llista.append(int(input()))
    elif opcio == 2:
        afegir = int(input())
        posicio = int(input())
        if posicio in range(0, len(llista)):
            llista.insert(posicio, afegir)
        else:
            print("Error: Posició no vàlida")
    elif opcio == 3:
        if len(llista) == 0:
            print("Error: Llista buida")
        else:
            llista.pop()
    elif opcio == 4:
        posicio = int(input())
        if posicio in range(0, len(llista)):
            llista.pop(posicio)
        else:
            print("Error: Posició no vàlida")
    elif opcio == 5:
        valor = str(input())
        if valor not in llista:
            print("Error: Valor inexistent a la llista")
        else:
            posicio = llista.index(valor)
            llista.pop(posicio)
    else:
        print("Error: Opció no disponible")
    
    print(llista)

    print("1-Afegir element al final la llista\n2-Afegir element en una posició de la llista\n3-Eliminar el darrer element de la llista\n4-Eliminar l'element d'una posició de la llista\n5-Eliminar la primera aparició d'un valor a la llista\n6-Sortir")
    opcio = int(input())

print(llista)