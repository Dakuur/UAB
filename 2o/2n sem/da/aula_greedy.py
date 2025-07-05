def ordenar_productes(productes: list, temps: list):
    return sorted(productes, key=lambda x: temps[x])

#cada índex de temps_llista és un codi de producte:
temps_llista = [
    6,
    1,
    3,
    2
]

prods = [0,0,1,2,1,3,3,2,1]

print(ordenar_productes(prods, temps_llista))
# resultat: [1, 1, 1, 3, 3, 2, 2, 0, 0] (codis de productes ordenats per temps d'entrega)

"""
La meva estratègia consisteix en ordenar la llista de productes
segons els temps d'encàrrec per a cada producte en la llista (temps_llista).
El resultat és una nova llista que conté els mateixos elements que la llista original,
però ordenats segons els temps d'encàrrec de manera ascendent.

Amb aquest mètode greedy, trobem una solució al problema, entregar primer els productes
que menys temps triguen. No és la millor solució pero compleix amb els requisits dels
algoritmes greedy.
"""