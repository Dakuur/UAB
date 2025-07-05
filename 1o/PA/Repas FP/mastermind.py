from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import random

@dataclass
class Codi:
    num_valors: int
    llista_valors: list = field(default_factory = list)

    def genera(self, colors_disponibles: dict):
        possibles = []
        for i in colors_disponibles.keys():
            possibles.append(i)
        sortida = random.choices(possibles, k = self.num_valors)
        return sortida
    
    def get_valor(self, posicio):
        return self.llista_valors[posicio]

    def indexs_valor(self, valor):
        llista = []
        for i in range(0, len(self.llista_valors)):
            if self.llista_valors[i] == valor:
                llista.append(i)
        return llista

@dataclass
class Intent:

    def __init__(self, num_valors: int):
        #combinacio: list = field(default_factory = list)
        #resultat: list = field(default_factory = list)
        self.num_valors = num_valors
        self.combinacio = []
        self.resultat = []
        self.valid = False
    
    def llegeix(self, colors_disponibles: Dict):

        entrada = input(f"Introdueix una combinació de {self.num_valors} colors: ")
        combinacio = [*entrada]
        if len(combinacio) != self.num_valors:
            self.valid = False
            return False
        for i in combinacio:
            if i not in colors_disponibles.keys():
                self.valid = False
                return False
        else:
            self.combinacio = combinacio
            self.valid = True
            return True
        
    '''
    def comprova(self, codi_secret: Codi):
        #combinació = codi = ["A", "B", "C", "D", "E"]
        self.resultat = []
        for i in range(0, self.num_valors):
            if self.combinacio[i] == codi_secret.llista_valors[i]:
                #self.resultat.append(1) #negre
                codi_secret.llista_valors[i] = "X"
            elif self.combinacio[i] in codi_secret.llista_valors:
                self.resultat.append(0) #blanc
                #codi_secret.llista_valors[i] = "X"
            print(codi_secret.llista_valors)
        self.resultat = sorted(self.resultat, reverse = True)
        return None
    '''

    def comprova(self, codi_secret: Codi):
        #combinació = codi = ["A", "B", "C", "D", "E"]
        resultat = []
        codi = codi_secret.llista_valors
        codi_copia = codi_secret.llista_valors.copy()
        combinacio = self.combinacio
        num = self.num_valors
        uniques = []
        for i in range(0, num):
            if combinacio[i] == codi[i]: 
                resultat.append(1) #negre, en el lloc
                indextreure = codi_copia.index(combinacio[i])
                codi_copia.pop(indextreure)
        for i in range(0, num):
            if (combinacio[i] in codi_copia) and (combinacio[i] not in uniques): #not 1
                if combinacio[i] != codi[i]:
                    uniques.append(combinacio[i])
                    resultat.append(0) #blanc, no en lloc
        self.resultat = sorted(resultat, reverse = True)
        return None

    def get_combinacio(self):
        return self.combinacio

    def get_resultat(self):
        return self.resultat

    def get_correcte(self):
        return self.valid

    def encertat(self):
        if self.resultat == [1]*self.num_valors:
            return True
        else:
            return False

def mastermind(num_valors: int, intents: int, codi_secret: Codi, colors_disponibles: dict):
    llista_intents = []
    encertat = False
    while len(llista_intents) < intents:
        guess = Intent(num_valors)
        format = guess.llegeix(colors_disponibles)
        if format == True:
            guess.comprova(codi_secret)
        else:
            guess.resultat = "ERROR"
        print(guess.get_resultat())
        llista_intents.append(guess)
        if guess.encertat():
            encertat = True
            break
    print(guess.resultat, guess.valid)
    return encertat, llista_intents

colors_disponibles = {
    'V':'Vermell',
    'B':'Blau',
    'T': 'Taronja',
    'R':'Rosa',
    'E':'Verd',
    'M':'Marro',
    'G':'Groc',
    'N':'Negre'}

codi_secret = Codi(5, ['V', 'V', 'V', 'V', 'V'])

mastermind(5, 10, codi_secret, colors_disponibles)