# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 18:20:09 2019

@author: ernest
"""

class Jugador:
    def __init__(self):
        self._casella = 1
        self._pot_tirar = True
        self._n_torns_inactiu = 0
        self._guanyador = False
    
    def posicio(self):
        return self._casella
    
    def mou(self, casella):
        self._casella = casella
    
    def set_inactiu(self, n_torns):
        self._pot_tirar = False
        self._n_torns_inactiu = n_torns
    
    def pot_tirar(self):
        if not self._pot_tirar:
            if self._n_torns_inactiu == 0:
                self._pot_tirar = True
            else:
                self._n_torns_inactiu = self._n_torns_inactiu - 1
        return self._pot_tirar

    def guanya(self):
        self._guanyador = True