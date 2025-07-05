def file2list(nomfitxer):
    try:
        linies = open(nomfitxer, "r")               #linies com a arxiu obert
    except:
        raise FileNotFoundError
    else:
        llista = []                                 #Funció principl dins de else del try
        for i in linies:                            #for in linies
            llista.append(i[:-1])                   #i[:-1] per no agafar \n
        linies.close()
        return llista

def extract_element(phrase, position):
    llista = phrase.split(" ")                      #separar en espais i posar en llista
    if position not in range(0, len(llista)):
        raise IndexError("Position fora de rang")   #IndexError
    else:
        return(llista[position])

def histogram(dictionary, key):
    if key not in dictionary.keys():
        dictionary[key] = 1
    else:
        dictionary[key] += 1

def add_dictionary(dictionary, key, value):
    if key not in dictionary.keys():
        dictionary[key] = value
    else:
        dictionary[key].add(value)                  #.add per conjunts (valors no repetits)

def maximum_value (dictionary):
    llista_max=dictionary.values()
    m=max(llista_max)
    return(m)

def equal_value(dic, value):
    llistaclaus =[]
    for i in dic:
        if dic[i] == value:
            llistaclaus.append(i)
    return llistaclaus



dic = {
    "a": 1,
    "b": 25,
    "c": 4,
    "f": 4,
    "hola": 1,
    "2": 1
}

print(equal_value(dic, 1))