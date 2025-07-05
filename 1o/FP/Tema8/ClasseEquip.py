class Equip:
    def __init__(self, nom, ciutat, punts, pressupost):
        self.nom = nom
        self.ciutat = ciutat
        self.punts = punts
        self.pressupost = pressupost

    def __str__(self):
        return f"Equip: {self.nom} {self.ciutat} {self.punts} {self.pressupost}"

    def __lt__(self, altreequip):
        return self.punts < altreequip.punts

    def __gt__(self, altreequip):
        return self.punts > altreequip.punts

    def __le__(self, altreequip):
        return self.punts <= altreequip.punts

    def __ge__(self, altreequip):
        return self.punts >= altreequip.punts

    def __ne__(self, altreequip):
        return self.punts != altreequip.punts

    def __eq__(self, altreequip):
        return self.punts == altreequip.punts