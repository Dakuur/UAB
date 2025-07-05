import LlistaDobleN

def testInsereix():
    N_PROVES = 8
    reduccio = 0

    valorsLlista=[[ 0, 2, 4, 6 ],[ 0, 2, 4, 6 ],[ 0, 2, 4, 6 ],[ 0, 2, 4, 6 ],[ 0, 2, 4, 6 ],[ 0 ],[ 0 ],[] ]
    
    posicioInserir = [ 0, 1, 2, 3, -1, 0, -1, -1 ]
    valorInserir = 1
    resultatLlista =[ [ 1, 0, 2, 4, 6 ],[ 0, 1, 2, 4, 6 ],[ 0, 2, 1, 4, 6 ],[ 0, 2, 4, 1, 6 ],[ 0, 2, 4, 6, 1 ],[ 1, 0 ],[ 0, 1 ],[ 1 ] ]
    
    
    print( "Comment :=>> Iniciant test metode INSEREIX" )
    print( "Comment :=>> =============================" )
    
    for i in range(0, N_PROVES):
        print( "Comment :=>> TEST " , i + 1 )

        print( "Comment :=>> Creant Llista inicial................ " )
        print( "Comment :=>> Valors de la llista: ",  valorsLlista[i])
        try:		
            l=LlistaDobleN.LlistaDobleN(valorsLlista[i])
            print( "Comment :=>> Afegint element....................... " )
            print( "Comment :=>> Valor a inserir: " , valorInserir )
            print( "Comment :=>> Posició on inserir: " , posicioInserir[i] )
            print( "Comment :=>> ---" )
            print( "Comment :=>> Valors esperats de la llista final: ",resultatLlista[i])

            l.inserirPosicio( valorInserir,posicioInserir[i])
            print( "Comment :=>> ---" )
            print( "Comment :=>> Valors obtinguts de la llista final: ", l)

            valid = (l==resultatLlista[i])
            
            if (valid):
                print( "Comment :=>> CORRECTE" )
            else:
                print( "Comment :=>> ERROR" )
                reduccio += 1.0
        except:
            print( "Comment :=>> ERROR d'execucio: " )
            reduccio += 1.0
		
        print( "Comment :=>> -----------------------------------------------" )

    return reduccio

def testElimina():
    N_PROVES = 7
    reduccio = 0
    valorsLlista=[[0, 2, 4, 6 ],[0, 2, 4, 6 ],[0, 2, 4, 6 ],[0, 2, 4, 6 ],[0, 2, 4, 6 ],[0 ],[0 ]]
    
    posicioEliminar = [ 1 , 2, 3, 0, -1, 0, -1 ]
    resultatLlista = [[0, 4, 6 ],[ 0, 2, 6 ],[ 0, 2, 4 ],[ 2, 4, 6 ],[ 0,2, 4 ],[],[]]
    
	
    print( "Comment :=>> Iniciant test metode ELIMINA" )
    print( "Comment :=>> ============================" )
    
    for i in range(0, N_PROVES):
        print( "Comment :=>> TEST " , i + 1 )
        print( "Comment :=>> Creant Llista inicial................ " )
        print( "Comment :=>> Valors de la llista: ",valorsLlista[i])
        try:
            l=LlistaDobleN.LlistaDobleN(valorsLlista[i])
            print( "Comment :=>> Eliminant element....................... " )
            print( "Comment :=>> Posicio de l'element a eliminar: " , posicioEliminar[i] )
            print( "Comment :=>> ---" )
            print( "Comment :=>> Valors esperats de la llista final: ",resultatLlista[i])

            l.delete(l[posicioEliminar[i]])

            print( "Comment :=>> ---" )
            print( "Comment :=>> Valors obtinguts de la llista final: ",l)

			
            valid = l== resultatLlista[i]

            if (valid):
                print( "Comment :=>> CORRECTE" )
            else:
                print( "Comment :=>> ERROR" )
                reduccio += 1.0
			
        except:
            print( "Comment :=>> ERROR d'execucio: " )
            reduccio += 1.0
		
        print( "Comment :=>> -----------------------------------------------" )
	
    return reduccio

def testInsertList():
    N_PROVES = 8
    
    reduccio = 0
    valorsLlista=[[0, 2, 4, 6 ],[ 0, 2, 4, 6 ],[ 0, 2, 4, 6 ],[ 0, 2, 4, 6 ],[ 0, 2, 4, 6 ],[ 0 ],[ 0 ],[]]
    
    posicioOnInserir = [ 0, 1, 2, 2, -1, -1, 0, 0 ]
    
    valorsInserir=[[1, 3, 5, 7 ],[1, 3, 5, 7 ],[1 ],[],[1, 3, 5, 7 ],[1, 3, 5, 7 ],[1, 3, 5, 7],[1, 3, 5, 7 ]]
    resultatLlista=[[1, 3, 5, 7, 0, 2, 4, 6 ],[0, 1, 3, 5, 7, 2, 4, 6 ],[0, 2, 1, 4, 6 ],[0, 2, 4, 6 ],[0, 2, 4, 6, 1, 3, 5, 7 ],[0,1, 3, 5, 7 ],[1, 3, 5, 7,0 ],[1, 3, 5, 7 ]]
    

    print( "Comment :=>> Iniciant test metode INSERTLIST" )
    print( "Comment :=>> ===============================" )

    for i in range(0, N_PROVES):
        print( "Comment :=>> TEST " , i + 1 )

        print( "Comment :=>> Creant Llista inicial................ "  )
        print( "Comment :=>> Valors de la llista: ",  valorsLlista[i])
        try:
            l=LlistaDobleN.LlistaDobleN(valorsLlista[i])
            print( "Comment :=>> Valors a inserir: ",valorsInserir[i])
            print( "Comment :=>> Posicio on inserir: " , posicioOnInserir[i] )

            print( "Comment :=>> Inserint elements a la llista....................... " )
            print( "Comment :=>> ---" )
            print( "Comment :=>> Valors esperats de la llista final: ",resultatLlista[i])

            l.inserirPosicioList(posicioOnInserir[i], valorsInserir[i])

            print( "Comment :=>> ---" )
            print( "Comment :=>> Valors obtinguts de la llista final: ",l)

            valid = l == resultatLlista[i]

            if (valid):
                print( "Comment :=>> CORRECTE" )
            else:
                print( "Comment :=>> ERROR" )
                reduccio += 1.0
			
        except:
            print( "Comment :=>> ERROR d'execucio: " )
            reduccio += 1.0
		
        print( "Comment :=>> -----------------------------------------------" )
    return reduccio

def testReverse():
    N_PROVES = 4
    
    reduccio = 0

    valorsLlista=[[ 0, 2, 4, 6 ],[ 0, 2 ],[0 ],[]]
    
    resultatLlista=[[ 6, 4, 2, 0 ],[2, 0 ],[0 ],[]]
    

    print( "Comment :=>> Iniciant test metode REVERSE" )
    print( "Comment :=>> ============================" )

    for i in range(0,N_PROVES):
        print( "Comment :=>> TEST " ,i + 1 )

        print( "Comment :=>> Creant Llista original................ " )
        print( "Comment :=>> Valors de la llista: ", valorsLlista[i])
        try:
            l1=LlistaDobleN.LlistaDobleN(valorsLlista[i])

            print( "Comment :=>> Invertint la llista.........." )
            print( "Comment :=>> ---" )
            print( "Comment :=>> Valors esperats de la llista final: ",resultatLlista[i])

            l1.reverse()
            print( "Comment :=>> ---" )
            print( "Comment :=>> Valors obtinguts de la llista final: ",l1)

            valid = l1== resultatLlista[i]

            if (valid):
                print( "Comment :=>> CORRECTE" )
            else:
                print( "Comment :=>> ERROR" )
                reduccio += 1.0
        except:
            print( "Comment :=>> ERROR d'execucio: " )
            reduccio += 1.0
        print( "Comment :=>> -----------------------------------------------" )
    return reduccio


def testOperadorAssignacio():
    reduccio = 0

    valorsLlista = [ 0, 2, 4, 6 ]


    print( "Comment :=>> Iniciant test operador ASSIGNACIO" )
    print( "Comment :=>> =================================" )

    
    try:
        print( "Comment :=>> Creant Llista inicial................ " )
        print( "Comment :=>> Valors de la llista list: ", valorsLlista)
        l1= LlistaDobleN.LlistaDobleN(valorsLlista)
        print( "Comment :=>> Valors de la llista LlistaDobleN: ", l1)
        
        print( "Comment :=>> Fent copia de la llista inicial amb operador assignacio................ " )
        l2 = LlistaDobleN.LlistaDobleN( l1)
        print( "Comment :=>> Modificant llista inicial................ " )
        
        l1.append(100)
        
        print( "Comment :=>> Valors la llista obtinguda amb la copia: ", l2)
      
        if (l2 == LlistaDobleN.LlistaDobleN(valorsLlista)):
            print( "Comment :=>> CORRECTE" )
        else:
            print( "Comment :=>> ERROR" )
            reduccio += 3.0		
        
    except:
        print("Comment :=>> ERROR d'execucio: ")
        reduccio += 3.0

    print( "Comment :=>> -----------------------------------------------" )
    return reduccio


valid = True
grade = 0

print("Grade :=>> " , grade)

reduccio = testInsereix()
if (reduccio > 3.0):
    reduccio = 3.0
grade += (2.0 - reduccio)
if (grade < 0):
    grade = 0.0

print("Grade :=>> " , grade)
		
reduccio = testElimina()
if (reduccio > 3.0):
    reduccio = 3.0
grade += (2.0 - reduccio)
if (grade < 0):
    grade = 0.0
print("Grade :=>> " , grade )

reduccio = testInsertList()
if (reduccio > 3.0):
    reduccio = 3.0
grade += (2.0 - reduccio)
if (grade < 0):
    grade = 0.0
print( "Grade :=>> " , grade )


reduccio = testReverse()
if (reduccio > 3.0):
    reduccio = 3.0
grade += (2.0 - reduccio)
if (grade < 0):
    grade = 0.0
print( "Grade :=>> " , grade)

reduccio = testOperadorAssignacio()
if (reduccio > 3.0):
    reduccio = 3.0
grade += (2.0 - reduccio)
if (grade < 0):
	grade = 0.0
print( "Grade :=>> " , grade )

if (grade < 0):
    grade = 0.0
print("Comment :=>> ------------------------------------------" )

if (grade == 10.0):
    print("Comment :=>> Final del test sense errors" )
print("Grade :=>> " , grade )
