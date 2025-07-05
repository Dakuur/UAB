# -*- coding: utf-8 -*-

from lliuramentEstudiant import LliuramentEstudiant
from activitat import Activitat
from datetime import date, today

class Lliurament(Activitat):
    def __init__(self, nom = "", descripcio = "", data_limit = date(1, 1, 1)):
        super().__init__(nom, descripcio)
        self._data_limit = data_limit
        self._lliuraments = []
    
    def visualitza(self, usr):
        assert today() < self._data_limit, "Fora de plaç"
        fitxer = input ("Nom del fitxer: ")
        self._lliuraments.append(LliuramentEstudiant(usr.niu, today(), fitxer))
    
    def avaula(self, usr):
        nota = float(input("Nota: "))
        usr.afegeix_nota(nota)
        
        
        
    

