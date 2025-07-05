def load_data(filename):
    filehandle=open(filename,"r")   #OBRIR ARXIU, NO LLEGIR! MODE "READ"
    dic=dict()                      #INICIALITZA DICCIONARI
    for line in filehandle:         #FOR AMB VARIABLE DE OPEN. NO LLEGIR!
        aux=line[:-1].split(',')    #LINE[:-1] PER NO LLEGIR \N. SPLIT AMB SEPARADOR
        dic[aux[0]]=aux[1:]         #CREA LA KEY DEL DICCIONARI I LA DEFINICIÓ. AUX[1:] PER NO AGAFAR LA KEY COM A ELEMENT DE LLISTA!
    filehandle.close()              #TANCAR ARXIU!
    return dic

def num_answer(dic, number, answer):
    contador = 0
    for estudiant in dic.values():  #DIC.VALUES()
        if estudiant[number - 1] == answer:
            contador = contador + 1
    return contador

def detect_copies(dic, niu1, niu2):
    iguals = 0
    resp1 = dic[niu1]                   #resp1 = dic[niu1]
    resp2 = dic[niu2]                   #resp2 = dic[niu2]
    i = 0
    while (i < 10) and (iguals < 7):    #(i < 10) and (iguals < 7)
        if resp1[i] == resp2[i]:
            iguals = iguals + 1
            i = i + 1
        return (iguals >= 7)            #return (iguals >= 7)

def get_grades(dic,solutions):
    notes=dict()
    for niu in dic.keys():
        notes[niu]=0
        for r1,r2 in zip(dic[niu],solutions):
            notes[niu]+=correccio(r1,r2)
    return notes

def correccio(r1,r2):
    if r1==r2:
        return 1
    elif r1=='x':
        return 0
    else:
        return -1/5

NUM_PREGUNTES=10
resp='S'     
while(resp=='S')or(resp=='s'):
    print("Introdueix les respostes correctes")
    correctes=list()
    for i in range(NUM_PREGUNTES):
        resp_correcta=input("Resposta pregunta "+str(i)+": ")
        correctes.append(resp_correcta)
    filename=input("Introdueix el nom del fitxer: ")
    dades=load_data(filename)
    print(dades)
    for preg in range(NUM_PREGUNTES):
        missatge="Pregunta "+str(preg+1)+"-> "
        for resp in ('a','b','c','d','e'):
            missatge+=resp+":"+str(num_answer(dades,preg,resp))+", "
        missatge+="NC"+":"+str(num_answer(dades,preg,'x'))
        print(missatge)
    nius=list()
    for x in dades.keys():
        nius.append(x)
    for i in range(len(nius)-1):
        for j in range(i+1,len(nius)):
            if detect_copies(dades,nius[i],nius[j])==True:
                print("NIUs "+str(nius[i])+" i "+str(nius[j])+"sospitosos de còpia")
    notes=get_grades(dades,correctes)
    for niu in notes.keys():
        print("NIU "+str(niu)+": "+str(notes[niu]))
    resp=input("Vols corregir un altre examen? ") 