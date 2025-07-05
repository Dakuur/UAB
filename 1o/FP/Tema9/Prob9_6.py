def file2list(filename):
    linies = open(filename, "r")
    list = []
    for linia in linies:
        list.append(linia[:-1])
    return list

def biseccio(llista, valor):
    esquerra = 0
    dreta = len(llista) - 1
    while esquerra <= dreta:
        mitjana = (esquerra + dreta) // 2
        if llista[mitjana] == valor:
            return True
        elif llista[mitjana] < valor:
            esquerra = mitjana + 1
        else:
            dreta = mitjana - 1
    return False

def parells_inversos(llista):
    parells = []
    for paraula in llista:
        paraula_inversa = paraula[::-1]
        if biseccio(llista, paraula_inversa):
            parells.append(paraula)
    return parells

list = file2list("catala.txt")

print(parells_inversos(list))