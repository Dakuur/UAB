def index_paraules(nom_fitxer):
    index = {}
    num_linia = 0
    with open(nom_fitxer, 'r') as fitxer:
        for linia in fitxer:
            paraules = linia.replace(',', ' ').replace('.', ' ').replace(':', ' ').replace(';', ' ').replace('!', ' ').replace('?', ' ').lower().split()
            for paraula in paraules:
                if paraula not in index:
                    index[paraula] = []
                index[paraula].append([num_linia, paraules.count(paraula)])
            num_linia += 1
    return index

def cerca_paraula(paraula, index, nomFitxer):
    resultat = index[paraula.lower()]   #[[1, 1], [3, 1]]   [[linia, frequencia]]
    fitxer = open(nomFitxer, "r")
    linies = fitxer.readlines()
    sortida = list()
    for element in resultat:    #element: [3, 1]   [linia, frequencia]
        res = [linies[element[0]][:-2], element[1]]
        if res not in sortida:
            sortida.append(res)
    fitxer.close()
    return sortida

index = index_paraules("test.txt")

print(cerca_paraula("is", index, "test.txt"))