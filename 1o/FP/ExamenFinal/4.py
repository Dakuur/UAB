class Influencer:
    def __init__(self,nom,num_seguidors,num_xarxes,nicks=[]):
        self.nom=nom
        self.num_seguidors=num_seguidors
        self.num_xarxes=num_xarxes
        self.nicks=nicks
    def __str__(self):
        sortida=self.nom+"( "
        for x in self.nicks[:-1]:
            sortida+=x+", "
        sortida+=self.nicks[-1:]+") "+self.num_seguidors
        return sortida     
    def __le__(self,other):
        if self.num_seguidors<=other.num_seguidors:
            return True
        else:
            return False
    def afegir_xarxa(self,seguidors_xarxa,nick_xarxa):
        if seguidors_xarxa>0:
            self.num_seguidors+=seguidors_xarxa
            self.num_xarxes+=1
            if nick_xarxa not in self.nicks:
                self.nicks.append(nick_xarxa)
        else:
            print("Error número de seguidors no vàlid")
    def mitjana_seguidors(self):
        return self.num_seguidors/self.num_xarxes