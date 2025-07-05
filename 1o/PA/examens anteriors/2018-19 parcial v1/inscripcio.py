# -*- coding: utf-8 -*-

from activitat import Activitat
from datetime import date, today
from grup import Grup


class Inscripcio(Activitat):
    def __init__(self, nom = "", descripcio = "", data_limit = date(1, 1, 1), n_grups = 0, n_estudiants_grup = 0):
        super().__init__(nom, descripcio)
        self._data_limit = data_limit
        self._n_grups  = n_grups
        self._n_estudiants_grup = n_estudiants_grup
        self._grups = {}
    
    def crea_grups(self):
        for i in range(self.n_grups):
            self._grups[i] = Grup(i)
            
    def visualitza(self, usr):
        assert today() < self._data_limit, "Fora de plaç"
        grup = input ("Codi del grup: ")
        assert len(self._grups[grup]) < self.n_estudiants_grup, "Grup ple"
        for g in self._grups.values:
            assert not usr.niu in g.estudiants
        self._grups[grup].afegeix_estudiant(usr.niu)
            