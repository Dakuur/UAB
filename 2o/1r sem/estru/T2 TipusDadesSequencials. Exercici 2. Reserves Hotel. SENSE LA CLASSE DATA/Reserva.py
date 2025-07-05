from dataclasses import dataclass, field
import datetime

@dataclass
class Reserva:
    nomClient: str
    dataEntrada: datetime.datetime
    dataSortida: datetime.datetime
    nHabitacions: int
    preu: float

    def __str__(self):
        return f'{self.nomClient} {self.dataEntrada} {self.nHabitacions}'