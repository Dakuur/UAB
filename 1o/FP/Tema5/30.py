def LlegirLlista(elements):
    llista = []
    for i in range(0, elements):
        llista.append(int(input()))
    return llista

def SumarLlistes(llista1, llista2):
    suma = []
    for i in range(0,len(llista1)):
        suma.append(llista1[i]+llista2[i])
    return suma

elements = int(input())

llista1 = LlegirLlista(elements)
llista2 = LlegirLlista(elements)

suma = SumarLlistes(llista1,llista2)

print(suma)