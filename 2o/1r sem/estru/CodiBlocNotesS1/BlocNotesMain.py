import BlocNotes
#import copy

#Primer main
print("====CREEM b com a Bloc Notes i afegim 2 notes===")
b=BlocNotes.BlocNotes()
b.add("N1","Dema classe ED")
b.add("N2","Comprar memoria")

nota = b.getNota("N1")
print("===La nota N1 es:",nota)

print("El bloc b es:")
print(b)

b.elim("N1")

print("El bloc b despres d’eliminar N1 es:")
print(b)


'''
#Segon main
grade =0.0
print("Comment :=>>CREEM b com a Bloc Notes i afegim 2 notes")
b=BlocNotes.BlocNotes()
grade+=1
b.add("N1","Dema classe ED")
b.add("N2","Comprar memoria")
grade+=1  
nota, existeix = b.getNota("N1")
if (existeix):
  grade+=1
  if (nota.missatge=="Dema classe ED" ):
    print("Comment :=>> La nota N1 es:",nota.missatge)
    grade+=1
  else:
    print("Comment :=>>Error:",nota,":CORRECTA: Dema classe ED ")
else:
    print("Comment :=>>Error: N1 no existeix:CORRECTA N1 existeix i val: Dema classe ED ")

b.elim("N1")
print("El bloc b despres d’eliminar N1 es:")
print(b)
grade+=1

if (grade < 0):
    grade = 0.0
print("Comment :=>> ------------------------------------------" )
if (grade == 5.0):
    print("Comment :=>> Final del test sense errors" )
print("Grade :=>> " , grade )
'''