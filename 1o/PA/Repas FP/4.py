def interseccio(llista1, llista2):
    interseccio = []
    for i in llista1:
        if i in llista2:
            interseccio.append(i)
    return interseccio