# -*- coding: utf-8 -*-
from usuari import Usuari

class Professor(Usuari):
    def __init__(self, niu, nom, mail):
        super().__init__('P', niu, nom, mail)
        self._activitats = []
    
    def registra_activitat(self, nom):
        self._activitats.append(nom)
    
    def consulta_activitat(self, nom):
        return nom in self._activitats
