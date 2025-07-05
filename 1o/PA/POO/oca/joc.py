from typing import List, Dict, Tuple
from dataclasses import dataclass, field

@dataclass
class Jugador:
    posició: int = field(default = 0)
    #pot_tirar: bool = field(default = True)
    n_torns_inactiu: int = field(default = 0)
    es_guanyador: bool = field(default = False)
    en_oca: bool = field(default = False)

@dataclass
class Casella:
    pass

@dataclass
class Normal(Casella):
    def ex_acc(self, J: Jugador):
        return J

@dataclass
class Oca(Casella):
    def ex_acc(self, J: Jugador):
        J.en_oca = True
        return J

@dataclass
class Pou(Casella):
    def ex_acc(self, J: Jugador):
        J.n_torns_inactiu = 2
        return J

@dataclass
class Mort(Casella):
    def ex_acc(self, J: Jugador):
        J.posició = 0
        return J

@dataclass
class Final(Casella):
    def ex_acc(self, J: Jugador):
        J.es_guanyador = True
        return J

@dataclass
class Tauler:
    caselles: List[Casella] = field(default = List[Casella])
    jugadors: List[Jugador] = field(default = List[Jugador])
    jugador_actual: int = field(default = -1)

    def inicialitza(self, nom_fitxer: str, n_jugadors: int):
        file = open(nom_fitxer, "r")
        for i in file:
            if i[-1] == "\n":
                i = i[:-1].split()
            else:
                i = i.split()
            tipo = i[1]
            if tipo == "N":
                casella = Normal()
            elif tipo == "O":
                casella = Oca()
            elif tipo == "P":
                casella = Pou()
            elif tipo == "M":
                casella = Mort()
            else:
                casella = Final()
            self.caselles.append(casella)
        file.close()

        for i in range(0, n_jugadors):
            jug = Jugador()
            self.jugadors.append(jug)
        
        self.jugador_actual = 0

        return None

    def torn_joc(self, valor_dau: int):
        i = self.jugador_actual
        jugador = self.jugadors[i]
        if jugador.posició + valor_dau >= len(self.caselles):
            return None
        self.jugadors[i].posició += valor_dau
        x = jugador.posició
        self.jugadors[i] = self.caselles[x].ex_acc(self.jugadors[i])

    def guanyador(self):
        for i in range(0, len(self.jugadors)):
            if self.jugadors[i].es_guanyador:
                return i
        return False

    def canvia_torn(self):
        i = self.jugador_actual
        if i >= len(self.jugadors) - 1:
            self.jugador_actual = 0
        else:
            self.jugador_actual = i + 1
        return None

def joc_oca(nom_fitxer: str, n_jugadors: int, valors_dau: List[int]):
    tauler = Tauler([], [], -1)
    tauler.inicialitza(nom_fitxer, n_jugadors)
    num_dau = 0

    torns = []

    while tauler.guanyador() == False:
        index = tauler.jugador_actual
        if tauler.jugadors[index].n_torns_inactiu > 0: #no pot tirar
            tauler.jugadors[index].n_torns_inactiu -= 1
            torns.append((index+1, False, tauler.jugadors[index].posició + 1))
            num_dau += 1 #ignorar valor dau
        else:
            tauler.torn_joc(valors_dau[num_dau])

            if tauler.jugadors[index].en_oca and Oca() in tauler.caselles[tauler.jugadors[index].posició:]:
                #va a seguent oca
                x = tauler.jugadors[index].posició + 1
                while True:
                    if type(tauler.caselles[x]) == Oca:
                        break
                    x += 1
                tauler.jugadors[index].posició = x

                torns.append((index+1, True, tauler.jugadors[index].posició + 1))

                num_dau += 1
                tauler.torn_joc(valors_dau[num_dau])
                tauler.jugadors[index].en_oca = False

                torns.append((index+1, True, tauler.jugadors[index].posició + 1))

            else:
                torns.append((index+1, True, tauler.jugadors[index].posició + 1))
            num_dau += 1
        tauler.canvia_torn()
    return (tauler.guanyador() + 1), torns