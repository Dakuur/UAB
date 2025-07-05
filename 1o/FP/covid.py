from time import time

inici = time()

fitxer = open("Covid Data.csv","r")

linies = fitxer.readlines()

for i in range(0, 1):
    print(linies)

final = time()

print("Temps total:", str(final - inici))