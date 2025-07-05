any = int(input())

dies = 28

if any%4 == 0:
    dies = 29
if any%100 == 0:
    dies = 28
if any%400 == 0:
    dies = 29

print("A l'any", str(any) + ", febrer té", str(dies), "dies")