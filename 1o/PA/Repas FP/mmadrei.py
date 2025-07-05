from dataclasses import dataclass, field
from typing import List, Dict, Tuple, ClassVar
import random

@dataclass
class Codi:
    valors_codi : int
    llista_valors : List = field(default_factory=list)
    
    def genera(self, colors_disponibles):
        llista = []
        for x in colors_disponibles.keys():
            llista.append(x)
        codi = random.choices(llista, k = self.valors_codi)
        self.llista_valors = codi
        return codi
    
    def get_valor(self, posicio):
        return self.llista_valors[posicio]
    
    def indexs_valor(self, valor):
        llista_indexs = []
        for x, y in enumerate (self.llista_valors):
            if y == valor:
                llista_indexs.append(x)
        return llista_indexs
    
    
    
@dataclass
class Intent:
    def __init__(self, valors_codi):
        self.valors_codi = valors_codi
        self.valors_player = []
        self.resultat_comparacio = []
        self.valid = True
        encertat : bool = field(init=False, default=False)
        
    def llegeix(self, colors_disponibles):
        combinacio = str(input("Introdueix valors de la combinació: "))
        llista_combinacio = [*combinacio]
       
        if len(llista_combinacio) != self.valors_codi:
            self.valid = False
            return False
 
        
        
        
        for x in llista_combinacio:
            if x not in colors_disponibles.keys():
                self.valid = False
                return False
        
        self.valors_player = llista_combinacio
        return True

        
    def comprova(self, codi_secret):
        no_repeticio = []
        llista = codi_secret.llista_valors.copy()
        for i in range (self.valors_codi):
            
            
            if self.valors_player[i] == codi_secret.llista_valors[i]: 
                
                self.resultat_comparacio.append(1)
                
                x = llista.index(self.valors_player[i])
                llista.pop(x)
                
             
        for i in range (self.valors_codi):
                
            if (self.valors_player[i] in llista) and (self.valors_player[i] not in no_repeticio) and (self.valors_player[i] != codi_secret.llista_valors[i]):
              
                no_repeticio.append(self.valors_player[i])
                self.resultat_comparacio.append(0)
                
    def get_combinacio(self):
        return self.valors_player
    
    def get_resultat(self):
        return self.resultat_comparacio
        
    def get_correcte(self):
        return self.valid
    
    def encertat(self):
        if self.resultat_comparacio == [1]*self.valors_codi:
            return True
        else:
            return False
         
                
        
def mastermind(n_valors,n_intents,codi_secret,colors_disponibles):
    llista_combinacions = []
    iterable = 1
    guess = False
    
    
    while (iterable <= n_intents) and (guess == False):
        attempt = Intent(n_valors)
        status = attempt.llegeix(colors_disponibles)
        
        if status == True:
        
            attempt.comprova(codi_secret)
        
        else:
            attempt.resultat_comparacio = "ERROR"
          
        llista_combinacions.append(attempt)
        print(attempt.get_resultat())
        if attempt.encertat() == True:
            guess = True
        else:
            guess = False
        
        iterable += 1
            
    if guess == True:
        return 1, llista_combinacions
    else:
        return 0, llista_combinacions
    
colors_disponibles = {
    'V':'Vermell',
    'B':'Blau',
    'T': 'Taronja',
    'R':'Rosa',
    'E':'Verd',
    'M':'Marro',
    'G':'Groc',
    'N':'Negre'}

codi_secret = Codi(5, ['V', 'B', 'E', 'B', 'V'])

mastermind(5, 10, codi_secret, colors_disponibles)