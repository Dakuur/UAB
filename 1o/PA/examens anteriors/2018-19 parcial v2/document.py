# -*- coding: utf-8 -*-


from activitat import Activitat

class Document(Activitat):
    def __init__(self, nom = "", descripcio = "", fitxer = ""):
        super().__init__(nom, descripcio)
        self._fitxer = fitxer
        
    @property
    def fitxer(self):
        return self._fitxer
    @fitxer.setter
    def fitxer(self, valor):
        self._fitxer = valor
        
    def visualitza(self, usr):
        print(self.fitxer)