from dataclasses import dataclass
import datetime

@dataclass
class Reserva:
    nomClient: str
    dataEntrada: datetime
    dataSortida: datetime
    nHabitacions: int
    preu: float

    def __str__(self):
        return f'{self.nomClient} {self.dataEntrada} {self.nHabitacions}'

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
                info = linia.removesuffix("\n").split()
                nomClient=info[0]
                entrada = datetime(int(info[1]), int(info[2]), int(info[3]))
                n_dies = int(info[5])
                hab = int(info[4])
                self.afegeixReserva(nomClient, entrada, n_dies, hab)
        return None
    
    def nReservesDia(self, data: datetime):
        total = 0
        for i in self.llista_reserves:
            if i.dataEntrada <= data and i.dataSortida >= data:
                total += i.nHabitacions
        return total

    def afegeixReserva(self, nomClient, dataEntrada, nDies, nHabitacions):
        sortida = dataEntrada+datetime.timedelta(days=nDies)
        d = dataEntrada
        while d <= sortida:
            if nHabitacions + self.nReservesDia(d) > self.nHab:
                return False
            d += datetime.timedelta(days=1)
        reserva = Reserva(nomClient, dataEntrada=dataEntrada, dataSortida=sortida, nHabitacions=nHabitacions, preu=self.preu_dia*nDies)
        self.llista_reserves.append(reserva)
        return True

    def consultaReserva(self, nomClient, dataEntrada):
        for i in self.llista_reserves:
            if i.nomClient == nomClient and i.dataEntrada == dataEntrada:
                return True, i.dataSortida, i.nHabitacions, i.preu

    def __str__(self):
        print("hola")

a = ReservesHotel()
a.llegeixReserves("estru/reserves1.txt")
a.llegeixReserves("estru/reserves2.txt")
print(a.consultaReserva("CLIENT_1", datetime(2,1,2018)))