import random
from time import sleep

que = int(input("De l'1 al qué: "))

sleep(1)
print("3...")
sleep(1)
print("2...")
sleep(1)
print("1...")

resultat = random.randrange(1,que)

ya = int(input())

print(resultat)

diferencia = abs(resultat - ya)

if resultat == ya:
    print("PRINGAS")
elif diferencia/resultat < 0.1:
    print("Caaaaaaaaaaaasi")