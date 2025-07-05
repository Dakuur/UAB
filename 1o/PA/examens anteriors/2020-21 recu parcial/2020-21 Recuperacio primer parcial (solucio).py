# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:18:52 2021

@author: 1002245
"""
from dataclasses import dataclass

@dataclass
class Aula:

    _codi: str = ""
    _capacitat = 0
    _tipus = ''
    _camera = False
    _n_ordinadors = 0
    _reserves = []
    
    @property
    def codi(self):
        return self._codi
    
    @property
    def capacitat(self):
        return self._capacitat
    
    def llegeix(self):
        self._codi = input("Codi:")
        self._capacitat = int(input("Capacitat: "))
    
    def llegeix_classe(self):
        self._tipus = 'classe'
        camera = input("Camera (S/N)")
        if camera == 'S':
            self._camera = True
        else:
            self._camera = False
    
    def llegeix_laboratori(self):
        self._tipus = 'laboratori'
        self.n_ordinadors = int(input("N. ordinadors: "))
        
    def reserva_classe(self):
        pass
    
    def reserva_laboratori(self):
        pass

class Facultat:
    def __init__(self, nom):
        self._nom = nom
        self._aules = dict()
    
    @property
    def nom(self):
        return self._nom
    
    def get_aula(self, codi):
        return self._aules[codi]
    
    def afegeix_aula(self, aula):
        self._aules[aula.codi] = aula
        
class Universitat:
    def __init__(self):
        self._facultats = []
    
    def afegeix_aula(self):
        tipus = input("Tipus Aula:")
        a = Aula()
        a.llegeix()
        if tipus == 'Laboratori':
            a.llegeix_laboratori()
        else:
            a.llegeix_classe()
        facultat = input("Nom facultat: ")
        for f in self._facultats:
            if f.nom == facultat:
                f.afegeix_aula(a)
                break
    
    def afegeix_reserva(self, facultat, codi_aula, n_persones):
        for f in self._facultats:
            if f.nom == facultat:
                aula = f.get_aula(codi_aula)
                if aula.capacitat > n_persones:
                    if aula.tipus == 'laboratori':
                        aula.reserva_laboratori()
                    else:
                        aula.reserva_classe()
                break

si = Aula()

print(si.capacitat)