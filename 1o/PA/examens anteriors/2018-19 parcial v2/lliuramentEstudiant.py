from datetime import date


class LliuramentEstudiant:
    def __init__(self, niu = "", data = date(1, 1, 1), fitxer = ""):
        self._niu = niu
        self._data = data
        self._fitxer = fitxer
    
    @property
    def niu(self):
        return self._niu
    @niu.setter
    def niu(self, valor):
        self._niu = valor
    
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, valor):
        self._data = valor

    @property
    def fitxer(self):
        return self._fitxer
    @fitxer.setter
    def fitxer(self, valor):
        self._fitxer = valor

