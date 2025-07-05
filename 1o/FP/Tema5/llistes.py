def LlegirLlista(elements):
    llista = []
    for i in range(0, elements):
        llista.append(int(input()))
    return llista

def InicialitzarLlista(elements,parametre):
    llista = []
    for i in range(0,elements):
        llista.append(parametre)
    return llista

def MitjanaLlista(llista):
    suma = 0
    elements = len(llista)
    for i in range(0,len(llista)):
        suma = suma + llista[i]
    mitjana = suma/elements
    return mitjana

def MaximLlista(llista):
    i = 0
    indexmax = i
    for i in range(0,len(llista)):
        if llista[i] > llista[indexmax]:
            indexmax = i
    return indexmax

def MinimLlista(llista):
    i = 0
    indexmin = i
    for i in range(0,len(llista)):
        if llista[i] < llista[indexmin]:
            indexmin = i
    return indexmin

def MinimLlistaNoZero(llista):
    i = 0
    indexmin = i
    if llista[indexmin] == 0:
        llista[indexmin] = 9999
    totzeros = True
    for i in range(0,len(llista)):
        if llista[i] != 0:
            totzeros = False
    for i in range(0,len(llista)):
        if llista[i] < llista[indexmin] and llista[i] != 0:
            indexmin = i
    if totzeros == True:
        indexmin = -1
    return indexmin