# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:03:12 2021

@author: 1002245
"""

from abc import ABC, abstractmethod
import numpy as np

class Partida:
    def __init__(self):
        self._jugadors = [[], []]
        self._torn_actual = 0
    
    def _busca_fitxa(self, posicio):
        posicions = [f.posicio for f in self._jugadors[self._torn_actual]]
        if posicio in posicions:
            return posicions.index(posicio)
        else:
            return -1
    
    def inicialitza(self, nom_fitxer):
        with open(nom_fitxer, "r") as fitxer:
            for linia in fitxer:
                valors = linia.split()
                if int(valors[3]) == 0:
                    fitxa = FitxaNormal((int(valors[0]), int(valors[1])), int(valors[2]))
                else:
                    fitxa = Dama((int(valors[0]), int(valors[1])), int(valors[2]))
                self._jugadors[int(valors[2])].append(fitxa)
    
    def mou_fitxa(self, posicio_inicial, posicio_final):
        valid = False
        index = self._busca_fitxa(posicio_inicial)
        if index != -1:            
            valid = self._jugadors[self._torn_actual][index].mou(posicio_final)
        return valid
    
    def converteix_dama(self, posicio):
        valid = False
        index = self._busca_fitxa(posicio)
        if index != -1:  
            valid = True
            self._jugadors[self._torn_actual][index] = Dama(posicio, self._torn_actual)
        return valid
    
    def genera_tauler(self):
        tauler = np.zeros((8,8), dtype='int')
        for jugador in self._jugadors:
            for fitxa in jugador:
                posicio = fitxa.posicio
                if type(fitxa) == FitxaNormal:
                    tauler[posicio] = fitxa.jugador + 1
                else:
                    tauler[posicio] = -fitxa.jugador - 1
        return tauler
    

class Fitxa(ABC):
    def __init__(self, posicio = (0,0), jugador = 0):
        self._posicio = posicio
        self._jugador = jugador

    @property
    def posicio(self):
        return self._posicio
    @posicio.setter
    def posicio(self, posicio):
        self._posicio = posicio
    
    @property
    def jugador(self):
        return self._jugador
    @jugador.setter
    def jugador(self, jugador):
        self._jugador = jugador
    
    @abstractmethod
    def mou(self, posicio_final):
        raise NotImplementedError()
        
class FitxaNormal(Fitxa):
    def __init__(self, posicio = (0,0), jugador = 0):
        super().__init__(posicio, jugador)
        if self._jugador == 0:
            self._direccio = 1
        else:
            self._direccio = -1
    
    def mou(self, posicio_final):
        valid = False
        if posicio_final[0] >= 0 and posicio_final[0] <= 7 and posicio_final[1] >= 0 and posicio_final[1] <=7:
            desplacament = (abs(posicio_final[0] - self._posicio[0]), posicio_final[1] - self._posicio[1])
            if desplacament == (1, self._direccio):
                self._posicio = posicio_final
                valid = True
        return valid

class Dama(Fitxa):
    def __init__(self, posicio = (0,0), jugador = 0):
        super().__init__(posicio, jugador)
    
    def mou(self, posicio_final):
        valid = False
        if posicio_final[0] >= 0 and posicio_final[0] <= 7 and posicio_final[1] >= 0 and posicio_final[1] <=7:
            desplacament = (posicio_final[0] - self._posicio[0], posicio_final[1] - self._posicio[1])
            if abs(desplacament[0]) == abs(desplacament[1]):
                self._posicio = posicio_final
                valid = True
        return valid
    
def comprova_moviment(tauler, posicio_inicial, posicio_final):
    if posicio_final[0] > posicio_inicial[0]:
        i_files = np.arange(posicio_inicial[0] + 1, posicio_final[0] + 1)
    else:
        i_files = np.arange(posicio_inicial[0] - 1, posicio_final[0] - 1, -1)
    if posicio_final[1] > posicio_inicial[1]:
        i_cols = np.arange(posicio_inicial[1] + 1, posicio_final[1] + 1)
    else:
        i_cols = np.arange(posicio_inicial[1] - 1, posicio_final[1] - 1, -1)
    recorregut = tauler[i_files, i_cols]
    fitxes_cami = recorregut[recorregut != 0]
    if fitxes_cami.size == 0:
        return True
    else:
        return False
        
