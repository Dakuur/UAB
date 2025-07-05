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