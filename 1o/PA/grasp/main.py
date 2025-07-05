from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from time import time

@dataclass
class Player:
    name: str = ""
    position: int = 0

@dataclass
class Game:
    players: List[Player] = field(default_factory=list)
    current_player: int = -1 #Jugador (index) del moviment acutal
    jumps: Dict[int, int] = field(default_factory=dict) #Key: valor de casella. Value: destí del salt (serp o escala) (si existeix)
    tiles_num: int = 0 #numero de caselles
    dice_list: List[int] = field(default_factory=list) #llista de valors del dau
    plays_list: List[str] = field(default_factory=list)
    dice_i: int = 0

    def init(self, filename):
        with open(filename, 'r') as file:
            seccio_actual = None

            for linia in file:

                if linia[0] == "[":
                    seccio_actual = linia[:-1]
                else:
                    if seccio_actual == '[Jugadors]':
                        self.players.append(Player(linia[:-1], 0))
                    elif seccio_actual == '[Caselles]':
                        self.tiles_num = int(linia)
                    elif seccio_actual == '[Serps]':
                        tpl = tuple(linia[:-1].split(","))
                        self.jumps[int(tpl[0])] = int(tpl[1])
                    elif seccio_actual == '[Escales]':
                        tpl = tuple(linia[:-1].split(","))
                        self.jumps[int(tpl[0])] = int(tpl[1])
                    elif seccio_actual == '[Dau]':
                        self.dice_list = [int(num) for num in linia.split(',')]

    def change_turn(self):
        if self.current_player < len(self.players) - 1:
            self.current_player += 1
        else:
            self.current_player = 0

    def throw_dice(self):
        value = self.dice_list[self.dice_i]
        self.dice_i += 1
        return value

    def winner(self):
        if (self.players[self.current_player].position == self.tiles_num - 1):
            return self.current_player
        else:
            return False
        #Retorna l'índex de jugador si el jugador actual està a la ultima casella (guanyador)
        #si no ha guanyat ningú encara, retorna true

    def check_empty(self, index):
        for player in self.players:
            if player.position == index:
                return False
        return True

    def make_play(self):
        dice = self.throw_dice()
        i = self.current_player
        inicial = self.players[i].position
        if self.check_empty(self.players[i].position + dice) == False:
            self.players[i].position = 0
        else:
            if (self.players[i].position + dice <= self.tiles_num - 1): #mira si no es passa del límit del tauler
                self.players[i].position += dice #mou la fitxa depenent del valor de dau
                try: #mira si ha caigut en escala o serp
                    self.players[i].position = self.jumps[self.players[i].position] #aplica el bonus/castig
                except: #si no és una casella especial
                    pass #es queda igual (només amb el valor del dau)
        final = self.players[i].position
        self.plays_list.append(f"{self.players[i].name} {inicial} {dice} {final}")

    def play_game(self):
        while not self.winner():
            self.change_turn()
            self.make_play()


    def print_plays(self):
        [print(play) for play in self.plays_list]

    def print_winner(self):
        print("Guanyador: " + self.players[self.current_player].name)

def rungame():
    G = Game()
    G.init("PA/grasp/input.txt")
    G.play_game()
    G.print_plays()
    G.print_winner()

start = time()

if __name__ == "__main__":
    rungame()

end = time()

print(f"Time: {end-start}s")