def csvFile2dic(nomfitxer, separador):
    try:
        linies = open(nomfitxer, "r")
    except FileNotFoundError:
        raise FileNotFoundError("Fitxer no trobat")
    else:                                               #TRY, EXCEPT, ELSE
        diccionari = dict()
        for linia in linies:                            #FOR IN LINIES
            llista = linia[:-1].split(separador)
            diccionari[llista[0]] = llista[1:]          #FORMAT IMPORTANT
        linies.close()                                  #CLOSE
    return diccionari

#def dic2csvfile(nomfitxer, dic, char):






print(csvFile2dic("2021.txt", ";"))