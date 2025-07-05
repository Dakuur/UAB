from re import I
from wsgiref.handlers import IISCGIHandler


estudiants = int(input())

aprovats = []
suspesos = []

for num in range (1, estudiants+1):
    niu = str(input())
    nota = float(input())
    if nota >= 5:
        aprovats.append(niu)
    else:
        suspesos.append(niu)

print("Aprovats:", len(aprovats), aprovats)
print("Suspesos:", len(suspesos), suspesos)