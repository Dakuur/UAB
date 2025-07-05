class Pais():
    def __init__(self, nom, superficie, llistaministres):
        self.nom = nom
        self.superficie = superficie
        self.llistaministres = llistaministres
    
    def __str__(self):
        stringministres = ""
        for i in self.llistaministres:
            stringministres = stringministres + i + ", "
        return(f"{self.nom} (superfície: {self.superficie})\n    Govern: {stringministres[:-2]}")
    
mac = Pais("Macedonia", 2.05, ["PerroSanxe", "Ada Colau"])
print(mac)