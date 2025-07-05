# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:31:25 2019

@author: Ernest
"""

from tauler import Tauler

t = Tauler()
t.inicialitza("oca.txt", 3)
print("\n--------------\nTorn del jugador 0. Valor del dau: 2. Nova posicio: 3. Tipus: oca. Salta oca. Posicio final: 5. Torna a tirar")
t.torn_joc(2)
print("\n--------------\nTorn del jugador 0. Valor del dau: 2. Nova posicio: 7. Tipus: pou. Dos torns sense tirar")
t.torn_joc(2)
print("\n--------------\nTorn del jugador 1. Valor del dau: 5. Nova posicio: 6. Tipus: normal")
t.torn_joc(5)
print("\n--------------\nTorn del jugador 2. Valor del dau: 4. Nova posicio: 5. Tipus: oca. Salta oca. Posicio final: 10. Torna a tirar")
t.torn_joc(4)
print("\n--------------\nTorn del jugador 2. Valor del dau: 3. Nova posicio: 13. No pot tirar. Posicio mes gran que el final")
t.torn_joc(3)
print("\n--------------\nTorn del jugador 0. Valor del dau: 2. No pot tirar. Salta torn")
t.torn_joc(2)
print("\n--------------\nTorn del jugador 1. Valor del dau: 3. Nova posicio: 9. Tipus: mort. Posicio final: 1")
t.torn_joc(3)
print("\n--------------\nTorn del jugador 2. Valor del dau: 1. Nova posicio: 11. Tipus: normal. ")
t.torn_joc(1)
print("\n--------------\nTorn del jugador 0. Valor del dau: 2. No pot tirar. Salta torn")
t.torn_joc(2)
print("\n--------------\nTorn del jugador 1. Valor del dau: 1. Nova posicio: 2. Tipus: normal.")
t.torn_joc(1)
print("\n--------------\nTorn del jugador 2. Valor del dau: 2. Nova posicio: 13. No pot tirar. Posicio mes gran que el final")
t.torn_joc(2)
print("\n--------------\nTorn del jugador 0. Valor del dau: 1. Nova posicio: 8. Tipus: normal. ")
t.torn_joc(1)
print("\n--------------\nTorn del jugador 1. Valor del dau: 2. Nova posicio: 4. Tipus: normal. ")
t.torn_joc(2)
print("\n--------------\nTorn del jugador 2. Valor del dau: 1. Nova posicio: 12. Tipus: final. Jugador guanya")
t.torn_joc(1)
