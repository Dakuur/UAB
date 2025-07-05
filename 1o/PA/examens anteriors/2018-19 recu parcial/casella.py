# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 18:32:36 2019

@author: ernest
"""

class Casella:
    def __init__(self, posicio, tauler):
        self._posicio = posicio
        self._tauler = tauler
    
    def entra_jugador(self, jugador):
        jugador.mou(self._posicio)
        return False
               
    @property
    def posicio(self):
        return self._posicio
    
    def es_oca(self):
        return False


class CasellaOca(Casella):
    def __init__(self, posicio, tauler):
        super().__init__(posicio, tauler)
    
    def entra_jugador(self, jugador):
        super().entra_jugador(jugador)
        print("Aquesta casella es una oca. Saltem a la següent oca")
        self._tauler.salta_oca(jugador)
        return True

    def es_oca(self):
        return True
        
class CasellaPou(Casella):
    def __init__(self, posicio, tauler):
        super().__init__(posicio, tauler)
    
    def entra_jugador(self, jugador):
        super().entra_jugador(jugador)
        print("Aquesta casella es el pou. Estaras 2 torns sense jugar")
        jugador.set_inactiu(2)
        return False
    
class CasellaMort(Casella):
    def __init__(self, posicio, tauler):
        super().__init__(posicio, tauler)
    
    def entra_jugador(self, jugador):
        super().entra_jugador(jugador)
        print("Aquesta casella es la mort. Tornes a la casella inicial")
        jugador.mou(1)
        return False
    
class CasellaFinal(Casella):
    def __init__(self, posicio, tauler):
        super().__init__(posicio, tauler)
    
    def entra_jugador(self, jugador):
        super().entra_jugador(jugador)
        print("Has arribat al final. Has guanyat")
        jugador.guanya()
        return False
        


        