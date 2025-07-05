def csv2dic(string, separador):
    linies = open(string, "r")
    dic = dict()
    for linia in linies:
        llista = linia[:-1].split(separador)
        dic[llista[0]] = llista[1:]
    return dic

print(csv2dic("notes.txt",","))