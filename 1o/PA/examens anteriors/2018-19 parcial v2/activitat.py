# -*- coding: utf-8 -*-

from abc import abstractmethod, ABCMeta

class Activitat(metaclass = ABCMeta):
    def __init__(self, nom = "", descripcio = ""):
        self._nom = nom
        self._descripcio = descripcio
        
    @property
    def nom(self):
        return self._nom
    @nom.setter
    def nom(self, valor):
        self._nom = valor
        
    @property
    def descripcio(self):
        return self._descripcio
    @descripcio.setter
    def descripcio(self, valor):
        self._descripcio = valor
        
    @abstractmethod
    def visualitza(self, usr):
        raise NotImplementedError()