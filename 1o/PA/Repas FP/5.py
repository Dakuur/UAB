def multiplicacio_elements(llista1, llista2):
    sortida = []
    for x in llista1:
        temp = []
        for y in llista2:
            temp.append(x*y)
        sortida.append(temp)
    return sortida