nombre = 0
suma = 0

i = True

num = int(input())
if num == 0:
    print("Error: La seqüència és buida. No es pot calcular la mitjana.")
else:
    while i is True:
        
        if num == 0:
            i = False
        else:
            suma = suma + num
            nombre = nombre + 1
            num = int(input())
    
    mitjana = suma / nombre

    print("Mitjana dels nombres de la seqüència:", mitjana)