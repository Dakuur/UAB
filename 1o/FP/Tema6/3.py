fitxer = open("Fitxer.txt", "r")

fitxer.seek(18)

print(fitxer.read(6))

print(fitxer.tell()+1)