from usuari import Usuari


class Estudiant(Usuari):
    def __init__(self, niu, nom, mail):
        super().__init__('E', niu, nom, mail)
        self._notes = []
    
    def afegeix_nota(self, nota):
        self._notes.append(nota)
    
    def nota_mitjana(self):
        return sum(self._notes) / len(self._notes)
    

    
