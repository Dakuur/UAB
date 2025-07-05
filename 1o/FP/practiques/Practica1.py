def PresentacioJoc():
    print("Pedra, paper, tisores, llangardaix, Spock és un joc d'atzar ampliació del popular Pedra, paper, tisores")
    print("Creat per Sam Kass amb Karen Bryla http://www.samkass.com/theories/RPSSL.html")
    print("Popularitzat per Sheldon Cooper a la sèrie Big Bang Theory.")
    print("Es fa servir per solucionar una disputa entre Sheldon i Raj en el capítol The Lizard - Spock Expansion")
    print("\nEl joc és al millor de N partides on N és un nombre senar")

def Senar(num):
    if  num%2 == 0:
        return False
    else:
        return True
   
def LlegirSenar():
    num = int(input("Introdueix el nombre de partides(senar): "))
    res  = Senar(num)
    while res == False:
        print("ERROR: El nombre introduït és parell")
        num = int(input("Introdueix el nombre de partides(senar): "))
        res  = Senar(num)
    return num

def MenuRPSLS():
    print("Escull entre:")
    print("1 - Rock")
    print("2 - Paper")
    print("3 - Scissors")
    print("4 - Lizard")
    print("5 - Spock")  

def LlegirNombre(minim, maxim):
    num = int(input("Entra valor entre "+str(minim)+" i "+str(maxim)+": "))
    while num not in range(minim,maxim+1):
        print("ERROR: Valor fora de l'interval")
        num = int(input("Entra valor entre "+str(minim)+" i "+str(maxim)+": "))
    return num

def JocRPSLS(opcio1,opcio2):
    ROCK = 1
    PAPER = 2
    SCISSORS = 3
    LIZARDS = 4
    SPOCK = 5
    if opcio1 == opcio2:
        return 0
    if opcio1 == ROCK and (opcio2 == SCISSORS or opcio2 == LIZARDS):
        return 1
    if opcio1 == PAPER and (opcio2 == ROCK or opcio2 == SPOCK):
        return 1
    if opcio1 == SCISSORS and (opcio2 == PAPER or opcio2 == LIZARDS):
        return 1
    if opcio1 == SPOCK and (opcio2 == SCISSORS or opcio2 == ROCK):
        return 1
    if opcio1 == LIZARDS and (opcio2 == SPOCK or opcio2 == PAPER):
        return 1
    else:
        return 2

def MissatgeRPSLS(player1,player2):
    if player1 == player2:
        print("Empat!!! ")
    else:
        if player1 == 1:
            if player2 == 4:
                print("Rock crushes Lizard")
            else:
                print("Rock crushes Scissors")
        elif player1 == 2:
            if player2 == 1:
                print("Paper covers Rock")
            else:
                print("Paper disproves Spock")
        elif player1 == 3:
            if player2 == 2:
                print("Scissors cuts Paper")
            else:
                print("Scissors decapitates Lizard")
        elif player1 == 4:
            if player2 == 5:
                print("Lizard poisons Spock")
            else:
                print("Lizard eats Paper")
        else:
            if player2 == 1:
                print("Spock smashes Scissors")
            else:
                print("Spock vaporizes Rock")
                
def Convertidor(llista):   
    for i in range (len(llista)):
        if llista[i] == 1:
            llista[i]="Rock"
        if llista[i] == 2:
            llista[i]="Paper"
        if llista[i] == 3:
            llista[i]="Scissors"
        if llista[i] == 4:
            llista[i]="Lizard"
        if llista[i] == 5:
            llista[i]="Spock"
    return llista

import random

def main():

    PresentacioJoc()
    nom = str(input("Introduir nom: "))
    random.seed(nom)
    numsenar = LlegirSenar()
    
    llistaSheldon = []
    llistaPersona = []
    counter1 = 0
    counter2 = 0
    victoria = 1+int(numsenar/2) 
    
    while counter1<(victoria) and counter2<(victoria):
        sheldon = random.randint(1,5)
        MenuRPSLS()
        persona = LlegirNombre(1,5)
        res = JocRPSLS(persona,sheldon)
        if res == 0:
            MissatgeRPSLS(persona,sheldon)
        if res == 1:
            MissatgeRPSLS(persona,sheldon)
            print("Guanya",nom,"!!!")
            counter1+=1
        if res == 2:
            MissatgeRPSLS(sheldon,persona)
            print("Guanya Sheldon Cooper!!!")
            counter2+=1
        print("MARCADOR -- Sheldon",counter2,nom,counter1)
        llistaPersona.append(persona)
        llistaSheldon.append(sheldon)
    
    if counter1 >= victoria:
        print("El guanyador és",nom)
    
    if counter2 >= victoria:
        print("El guanyador és Sheldon")
    
    llista1 = Convertidor(llistaPersona)
    llista2 = Convertidor(llistaSheldon)
    print(llista2)
    print(llista1)

main()