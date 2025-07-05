def suma_acumulada(llista):
    res = []
    suma = 0
    for element in llista:
        suma += element
        res.append(suma)
    return res


print(suma_acumulada([1,2,3,4,5]))