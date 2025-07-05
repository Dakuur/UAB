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
                
    def afegeix_activitat(self, usr):
        assert usr.tipus == 'P', "usuari no valid"
        nom = input("Nom activitat: ")
        descripcio = input ("Descripcio activitat: ")
        tipus_activitat = input("Tipus activitat: ")
        if tipus_activitat == 'D':
            fitxer = input ("Nom del fitxer: ")
            act = Document(nom, descripcio, fitxer)
        elif tipus_activitat == 'I':
            day = int(input("Dia limit"))
            month = int(input("Mes limit"))
            year = int(input("Any limit"))
            n_grups = int(input("N. grups: "))
            n_estudiants = int(input("N. estudiants per grup: "))
            act = Inscripcio(nom, descripcio, date(year, month, day), n_grups, n_estudiants)
        self._activitats[nom] = act
        usr.registra_activitat(nom)
    
    def get_activitat(self, nom):
        assert nom in self._activitats
        return self._activitats[nom]
    
    def visualitza_activitat(self, nom, niu):
        assert nom in self._activitats
        assert niu in self._usuaris
        self._activitats[nom].visualitza(self._usuaris[niu])
        
        
            