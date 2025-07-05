from llibre import Llibre
from revista import Revista
from prestec import Prestec
from publicacio import Publicacio

from dataclasses import dataclass, field, InitVar
from datetime import date, datetime, timedelta
from typing import List, Dict


@dataclass
class Biblioteca:
    _publicacions = {}
    _prestecs = []

    def llegeixPublicacions(self, nom_fitxer: str):
        file = open(nom_fitxer, "r")
        data = file.readlines()
        i = 0
        while True and i < len(data) - 1:
            if data[i][:-1] == "L":
                codi = data[i+1][:-1]
                titol = data[i+2][:-1]
                autor = data[i+3][:-1]
                n_copies = data[i+4][:-1]
                n_dies = data[i+5][:-1]
                llibre = Llibre(codi, titol, autor, n_copies, n_dies)
                self._publicacions[codi] = llibre
                i += 6
            elif data[i][:-1] == "R":
                codi = data[i+1][:-1]
                titol = data[i+2][:-1]
                periodicitat = data[i+3][:-1]
                if data[i+4][-1] == "\n":
                    exemplars_temp = data[i+4][:-1].split()
                else:
                    exemplars_temp = data[i+4].split()
                exemplars = list()
                for x in range(0, len(exemplars_temp)):
                    exemplars.append(int(exemplars_temp[x]))
                revista = Revista(codi, titol, periodicitat, exemplars)
                self._publicacions[codi] = revista
                i += 5
            else:
                pass
        file.close()

    def presta(self, codi_usr: str, codi_pub: str, data_préstec: date, n_exemplar: int):
        try:
            pub = self._publicacions[codi_pub]
        except KeyError:
            return bool(False), data_préstec #codi no trobat
        
        if type(pub) == Llibre:
            if int(pub._n_copies) > 0:
                data_retorn = data_préstec + timedelta(days = int(pub._n_dies))
                prestec = Prestec(str(codi_usr), str(codi_pub), data_préstec, data_retorn, 0)
                self._prestecs.append(prestec)
                self._publicacions[codi_pub]._n_copies = int(self._publicacions[codi_pub]._n_copies) - 1

                return bool(True), data_retorn
            else:
                return bool(False), data_préstec
        elif type(pub) == Revista:
            if pub._periodicitat == "MENSUAL":
                data_retorn = data_préstec + timedelta(days = 30)
            elif pub._periodicitat == "TRIMESTRAL":
                data_retorn = data_préstec + timedelta(days = 90)
            elif pub._periodicitat == "ANUAL":
                data_retorn = data_préstec + timedelta(days = 365)
            try:
                index = pub._exemplars.index(int(n_exemplar))
            except:
                return bool(False), data_préstec #exemplar no trobat
            prestec = Prestec(codi_usr, codi_pub, data_préstec, data_retorn, n_exemplar)
            self._prestecs.append(prestec)
            self._publicacions[codi_pub]._exemplars.remove(n_exemplar)

            return bool(True), data_retorn
        else:
            return bool(False), data_préstec

    def retorna(self, codi_usr: str, codi_pub: str, n_exemplar: int, data_actual: date):
        trobat = False
        for index in range(0, len(self._prestecs)):
            prestec = self._prestecs[index]
            if (prestec._codi_publicacio == codi_pub) and (codi_usr == prestec._codi_usuari):
                trobat = True
                break
        if trobat == False:
            return bool(False), bool(False) #no està a la llista de prestecs
        
        pub = self._publicacions[codi_pub]
        if type(pub) == Llibre:
            self._prestecs.pop(index)
            self._publicacions[codi_pub]._n_copies += 1
            a_temps = False
            if data_actual < prestec._data_retorn:
                a_temps = True
            return bool(True), bool(a_temps)
            
        elif type(pub) == Revista:
            if n_exemplar in pub._exemplars:
                return bool(False), bool(False) #exemplar no prestat
            else:
                a_temps = False
                if data_actual < prestec._data_retorn:
                    a_temps = True
                self._publicacions[codi_pub]._exemplars.append(n_exemplar)
                return bool(True), bool(a_temps)