from usuari import Usuari


class Estudiant(Usuari):
    def __init__(self, niu, nom, mail):
        super().__init__('E', niu, nom, mail)
        self._notes = []
    
    

    
