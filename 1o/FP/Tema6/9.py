def generar_llistat_notes(nom_fitxer):
    fitxer = open(nom_fitxer, "r")
    data = fitxer.read()

    palabras = data.split()

    suma = 0
    num = 0

    for i in range(0, len(palabras)):
        niu = int(i)
        for x in range(0, 10):
            try:
                print(int(palabras[niu]), float(palabras[niu+1+x]))
                suma += float(palabras[niu+1+x])
                num += 1
                break
            except:
                i = i
    mitjana = suma/num
    print("Nota mitjana classe:", mitjana)

generar_llistat_notes("DadesEstudiants.txt")