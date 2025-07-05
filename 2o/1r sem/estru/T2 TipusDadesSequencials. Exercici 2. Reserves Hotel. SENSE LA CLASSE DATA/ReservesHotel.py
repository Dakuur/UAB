import Reserva
import copy
import datetime
from dataclasses import dataclass

@dataclass
class ReservesHotel:

    def __init__(self, nom_hotel="", preu_dia=0.0, nHab=0, reserves=None):
        self.nom_hotel = nom_hotel
        self.preu_dia = preu_dia
        self.nHab = nHab
        if reserves is None:
            self.llista_reserves = []
        else:
            self.llista_reserves = reserves.llista_reserves
            
    def llegeixReserves(self, nomFitxer):        
        with open(nomFitxer, "r") as file:
            for linia in file:
                if "\n" in linia:
                    linia = linia[:-1]
                info = linia.split()
                nomClient=info[0]
                entrada = datetime.datetime(int(info[3]), int(info[2]), int(info[1]))
                n_dies = int(info[5])
                hab = int(info[4])
                self.afegeixReserva(nomClient, entrada, n_dies, hab)
        return None
    
    def nReservesDia(self, data: datetime.datetime):
        total = 0
        for i in self.llista_reserves:
            if i.dataEntrada <= data and i.dataSortida > data:
                total += i.nHabitacions
        return total

    def afegeixReserva(self, nomClient, dataEntrada, nDies, nHabitacions):
        sortida = dataEntrada+datetime.timedelta(days=nDies)
        d = dataEntrada
        while d <= sortida:
            if self.nReservesDia(d) + nHabitacions > self.nHab:
                return False
            d += datetime.timedelta(days=1)
        reserva = Reserva.Reserva(nomClient, dataEntrada=dataEntrada, dataSortida=sortida, nHabitacions=nHabitacions, preu=self.preu_dia*nDies*nHabitacions)
        self.llista_reserves.append(reserva)
        return True

    def consultaReserva(self, nomClient, dataEntrada):
        for i in self.llista_reserves:
            if i.nomClient == nomClient and i.dataEntrada == dataEntrada:
                return True, i.dataSortida, i.nHabitacions, i.preu
        return False, None, None, None

    def __str__(self):
        print("hola")