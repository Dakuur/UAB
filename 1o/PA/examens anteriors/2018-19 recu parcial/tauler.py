# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 18:51:17 2019

@author: Ernest
"""

import casella as c
import jugador as j

class Tauler():
    def __init__(self):
        self._caselles = []
        self._jugadors = []
        self._jugador_actual = 0
    
    def inicialitza(self, nom_fitxer, nJugadors):
        with open(nom_fitxer) as fitxer:
            for linia in fitxer:
                valors = linia.split()
                if valors[1] == 'O':
                    nova_casella = c.CasellaOca(int(valors[0]), self)
                elif valors[1] == 'P':
                    nova_casella = c.CasellaPou(int(valors[0]), self)
                elif valors[1] == 'M':
                    nova_casella = c.CasellaMort(int(valors[0]), self)
                elif valors[1] == 'F':
                    nova_casella = c.CasellaFinal(int(valors[0]), self)
                else:
                    nova_casella = c.Casella(int(valors[0]), self)
                self._caselles.append(nova_casella)
        for i in range(nJugadors):
            self._jugadors.append(j.Jugador())
        self._jugador_actual = 0
        
    def salta_oca(self, jugador):
        oques = [c.posicio for c in self._caselles if c.es_oca() and c.posicio > jugador.posicio()]
        if (len(oques) > 0):
            print ("Saltem oca. Nova posicio: ", oques[0])
            jugador.mou(oques[0])
        
    def torn_joc(self, valor_dau):
        print ("\nJugador amb el torn: ", self._jugador_actual)
        print ("Valor de dau: ", valor_dau)
        jugador_torn = self._jugadors[self._jugador_actual]
        torna_tirar = False
        if jugador_torn.pot_tirar():
            nova_posicio = jugador_torn.posicio() + valor_dau
            if nova_posicio <= len(self._caselles):
                print ("Nova posicio: ", nova_posicio)
                torna_tirar = self._caselles[nova_posicio-1].entra_jugador(jugador_torn)
                if jugador_torn.guanya():
                    print ("Jugador guanya. Final partida")
            else:
                print ("No pot tirar: nova posicio mes gran que el final")
        else:
            print ("No pot tirar. Salta torn")
        if not torna_tirar:
            self._jugador_actual = (self._jugador_actual  + 1) % len(self._jugadors)
                
                    
        
