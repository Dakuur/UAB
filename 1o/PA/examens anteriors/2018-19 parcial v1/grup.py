# -*- coding: utf-8 -*-


class Grup:
    def __init__(self, codi = 0):
        self._codi = codi
        self._estudiants = []
    
    @property
    def codi(self):
        return self._codi
    @codi.setter
    def codi(self, valor):
        self._codi = valor
    
    @property
    def estudiants(self):
        return self._estudiants
    
    def afegeix_estudiant(self, niu):
        self._estudiants.append(niu)

g = Grup