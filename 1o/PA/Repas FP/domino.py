from typing import List, Tuple, Dict
from dataclasses import dataclass, field

@dataclass
class Jugador:
    fitxes: List[tuple] = list

@dataclass
class Partida:
    jugadors: List[Jugador] = field(default_factory=List[Jugador])
    jugador_actual: Jugador = field(default_factory=Jugador)
    valor_esq: int = field(default_factory=int)
    valor_dreta: int = field(default_factory=int)
    n_torns_passant: int = field(default_factory=int)
    fitxes_jugades: list = field(default_factory=list)

    def inicialitza(self, fitxes: List[tuple]):
        '''Inicialitza la partida al seu estat inicial: reparteix les fitxes
        inicials entre els jugadors i inicialitza el jugador actual al
        primer jugador.
        PARAMETERS:
        fitxes:List[Fitxa]
        Llista amb totes les fitxes del joc ordenades per repartir
        entre els jugadors.'''

        '''
        for i in range(0, len(self.jugadors)):
            self.jugadors[i].fitxes = [fitxes[i] for i in range(0+7*i,7+7*i)]
            '''
        
        for i in range(0, len(self.jugadors)):
            fitxesjugador = fitxes[0+7*i:7+7*i]
            self.jugadors[i] = Jugador(fitxesjugador)

        
        #self.jugadors[0].fitxes = fitxes[0:7]
        #self.jugadors[1].fitxes = fitxes[7:14]
        #self.jugadors[2].fitxes = fitxes[14:21]
        #self.jugadors[3].fitxes = fitxes[21:28]

        self.jugador_actual = self.jugadors[0]
        return None

    def juga_torn(self) -> tuple:
        """Selecciona una fitxa del jugador actual i la col·loca al tauler
        seguint les regles indicades a l'enunciat. Si el jugador no té
        cap fitxa per jugar incrementa el nº de torns amb algun jugador
        passant per poder detectar si la partida queda bloquejada
        PARAMETERS:
        RETURN:
        fitxa_actual: Fitxa
        Fitxa sel·leccionada i col·locada al tauler. Si el jugador
        no té cap fitxa per jugar es retorna la fitxa (-1, -1)"""

        i = self.jugadors.index(self.jugador_actual)

        if len(self.fitxes_jugades) == 0:
            fitxa = self.jugador_actual.fitxes[0]
            self.valor_esq = fitxa[0]
            self.valor_dreta = fitxa[1]
            self.jugador_actual.fitxes.remove(fitxa)
            self.jugadors[i] = self.jugador_actual
            return fitxa
        
        fet = False

        for fitxa in self.jugador_actual.fitxes:
            if fitxa[0] == self.valor_esq:
                self.valor_esq = fitxa[1]
                fet = True
                break
            elif fitxa[1] == self.valor_esq:
                self.valor_esq = fitxa[0]
                fet = True
                break
            elif fitxa[0] == self.valor_dreta:
                self.valor_dreta = fitxa[1]
                fet = True
                break
            elif fitxa[1] == self.valor_dreta:
                self.valor_dreta = fitxa[0]
                fet = True
                break

        if fet == True:
            self.jugador_actual.fitxes.remove(fitxa)
            self.jugadors[i] = self.jugador_actual
            self.n_torns_passant = 0
            return fitxa

        self.n_torns_passant += 1
        fitxa_invalida = tuple((-1,-1))
        return fitxa_invalida

    def canvia_torn(self):
        """Canvia el jugador que té el torn per jugar.
        PARAMETERS:
        RETURN:"""

        i = self.jugadors.index(self.jugador_actual)
        if i >= len(self.jugadors) - 1:
            self.jugador_actual = self.jugadors[0]
        else:
            self.jugador_actual = self.jugadors[i + 1]
        return None

    def guanyador(self):
        """Comprova si el jugador actual que té el torn ha guanyat la partida
        (s'ha quedat sense fitxes)
        PARAMETERS:
        RETURN:
        guanyador: int
        Si el jugador actual és el guanyador retorna el nº del jugador
        actual (entre 1 i 4). Si no retorna -1"""

        if len(self.jugador_actual.fitxes) == 0:
            i = self.jugadors.index(self.jugador_actual)
            return i + 1
        return -1

    def partida_bloquejada(self):
        """Comprova si la partida està bloquejada perquè els 4 jugadors han
        passat tots un torn sense poder jugar
        PARAMETERS:
        RETURN:
        bloquejada: bool
        Si han passat 4 torns seguits sense que cap jugador hagi
        pogut tirar fitxa"""

        if self.n_torns_passant == len(self.jugadors):
            return True
        else:
            return False

def juga_domino(fitxes_inicials):
    jugadors = [Jugador([])]*4
    joc = Partida(jugadors)
    joc.inicialitza(fitxes_inicials)
    joc.fitxes_jugades = []
    guanyador = -1
    while not joc.partida_bloquejada() and guanyador == -1:
        fitxa = joc.juga_torn()
        joc.fitxes_jugades.append(tuple(fitxa))
        guanyador = joc.guanyador()
        if guanyador == -1:
            joc.canvia_torn()
    return guanyador, joc.fitxes_jugades




fitxes = []
for i in range(7):
    for j in range(i, 7):
        fitxes.append(tuple((i, j)))

guanyador, fitxes_jugades = juga_domino(fitxes)

print("Guanyador: Jugador", guanyador)

print("Jugades:", len(fitxes_jugades))

print("Fitxes jugades:", fitxes_jugades)