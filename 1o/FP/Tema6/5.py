nomfitxer = str(input())

fitxer = open(nomfitxer, "r")

data = fitxer.read()

paraules = 0

inici = 0
finallinia = 0

for i in range(0, len(data)):
    if data[i] == "\n":
        finallinia = i
        for j in range(inici, finallinia):
            if data[j] == " ":
                paraules += 1
        print(data[inici:i], paraules+1)
        inici = i
        paraules = 0