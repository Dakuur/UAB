class Estudiant:
    def __init__(self, niu, nom, cognoms, nota):
        self.niu = niu
        self.nom = nom
        self.cognoms = cognoms
        self.nota = nota

class Assignatura:
    def __init__(self, codi, nom):
        self.codi = codi
        self.nom = nom
        self.llista = []
    def niuinllista(self, niu):
        for i in self.llista:
            if i.niu == niu:
                return True
        return False

def demanarassignatura():
    codi = input("Codi: ")
    nomassignatura = input("Nom: ")
    assignatura = Assignatura(codi, nomassignatura)
    return assignatura

def demanarestudiant():
    niu = input("NIU: ")
    while assignatura.niuinllista(niu) == True:
        print("Error: Estudiant existent")
        niu = int(input("NIU: "))
    nom = str(input("Nom: "))
    cognoms = str(input("Cognoms: "))
    correcte = False
    while correcte == False:
        try:
            nota = float(input("Nota: "))
        except ValueError:
            print("Error: Tipus d'entrada incorrecte")
        else:
            correcte = True
    estudiant = Estudiant(niu, nom, cognoms, nota)
    return estudiant

assignatura = demanarassignatura()

for i in range(0, 6):
    estudiant = demanarestudiant()