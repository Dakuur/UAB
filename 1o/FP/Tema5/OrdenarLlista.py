def OrdenarLlista(llista):
    for i in range(1, len(llista)):       #Per i des de 1 fins la longitud de la llista - 1
        for j in range(0, len(llista)-i): #Per j des de 0 fins la longitud de la llista - i -1
            if  llista[j] > llista[j+1]:    #Si llista[j] > llista[j+1]
                temporal = llista[j]        #Intercanviar llista[j] i llista[j+1]
                llista[j] = llista[j+1]
                llista[j+1] = temporal
    print("Llista ordenada:",llista)