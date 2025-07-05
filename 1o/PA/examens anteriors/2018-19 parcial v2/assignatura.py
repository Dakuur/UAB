# -*- coding: utf-8 -*-

from document import Document
from lliurament import Lliurament
from inscripcio import Inscripcio
from estudiant import Estudiant
from professor import Professor
from datetime import date

class Assignatura:
    def __init__(self, nom = ""):
        self._nom = nom
        self._activitats = {}
        self._usuaris = {}
        
    def llegeix_usuaris(self, nom_fitxer):
        with open(nom_fitxer) as fitxer:
            for linia in fitxer:
                valors = linia.split()
                if valors[0] == 'E':
                    usr = Estudiant(valors[0], valors[1], valors[2], valors[3])
                else:
                    usr = Professor(valors[0], valors[1], valors[2], valors[3])
                self._usuaris[usr.niu] = usr
                
    def get_activitat(self, nom):
        assert nom in self._activitats
        return self._activitats[nom]
    
    def visualitza_activitat(self, nom, niu):
        assert nom in self._activitats
        assert niu in self._usuaris
        self._activitats[nom].visualitza(self._usuaris[niu])
        
    def llegeix_activitats(self, nom_fitxer):
        with open(nom_fitxer) as fitxer:
            for linia in fitxer:
                valors = linia.split()
                if valors[0] == 'D':
                    act = Document(valors[1], valors[2], valors[3])
                else:
                    act = Lliurament(valors[1], valors[2], date(valors[5], valors[4], valors[3]))
                self._activitats[act.nom] = act        
        
            