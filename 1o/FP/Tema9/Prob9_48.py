def CrearIndex(nom_fitxer):
    index = {}
    with open(nom_fitxer, "r") as fitxer:
        linies = fitxer.readlines()
        for i, linia in enumerate(linies):
            paraules = linia.split()
            for paraula in paraules:
                paraula = paraula.lower()
                if paraula not in index:
                    index[paraula] = set()
                index[paraula].add(i+1)
    return index