# -*- coding: utf-8 -*-

class Usuari:
    def __init__(self, tipus = "", niu = "", nom = "", mail = ""):
        self._tipus = tipus
        self._niu = niu
        self._nom = nom
        self._mail = mail
    
    @property
    def tipus(self):
        return self._tipus
    @tipus.setter
    def tipus(self, valor):
        self._tipus = valor
    
    @property
    def niu(self):
        return self._niu
    @niu.setter
    def niu(self, valor):
        self._niu = valor
    
    @property
    def nom(self):
        return self._nom
    @nom.setter
    def nom(self, valor):
        self._nom = valor

    @property
    def mail(self):
        return self._mail
    @mail.setter
    def mail(self, valor):
        self._mail = valor