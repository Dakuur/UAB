from dataclasses import dataclass, field
from typing import List


@dataclass
class Partida():
    
    def inicialitza(self, fitxes):
        self.jugadors: List = []
        self.jugador = []
        self.jugador_actual = []
        for i in range(4):
            self.jugador.append(fitxes)
            for j in range(7):
                self.jugador.append(fitxes.pop(0))
            
            self.jugadors.append(self.jugador)
            self.jugadors[0] = self.jugador_actual
        
    def juga_torn(self):
        for fitxa in self.jugador_actual.fitxes:
            if fitxa[0] == self.valor_esq:
                self.valor_esq = fitxa[1]
                fet = True
                break
            elif fitxa[1] == self.valor_esq:
                self.valor_esq = fitxa[0]
                fet = True
                break
            elif fitxa[1] == self.valor_dreta:
                self.valor_dreta = fitxa[0]
                fet = True
                break
            elif fitxa[0] == self.valor_dreta:
                self.valor_dreta = fitxa[1]
                fet = True
                break

    def canvia_torn(self):
        self.jugador_actual = (self.jugador_actual +1) % 4
        
    def guanyador(self):
        for i,jugador in enumerate(self.jugadors):
            if not jugador.fitxes:
                return i+1
        return -1
        
    def partida_bloquejada(self):
        return True


def juga_domino(fitxes_inicials):
    joc = Partida()
    joc.inicialitza(fitxes_inicials)
    fitxes_jugades = []
    guanyador = -1
    while not joc.partida_bloquejada() and guanyador == -1:
        fitxa = joc.juga_torn()
        fitxes_jugades.append((fitxa.valor1, fitxa.valor2))
        guanyador = joc.guanyador()
        if guanyador == -1:
            joc.canvia_torn()
    return guanyador, fitxes_jugades

fitxes = []
for i in range(7):
    for j in range(i, 7):
        fitxes.append(tuple((i, j)))

guanyador, fitxes_jugades = juga_domino(fitxes)

print("Guanyador: Jugador", guanyador)

print("Jugades:", len(fitxes_jugades))

print("Fitxes jugades:", fitxes_jugades)