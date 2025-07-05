diccandicats = dict()
candidat = str(input())
while candidat != "-1":
    if candidat not in diccandicats:
        diccandicats[candidat] = 1
    else:
        diccandicats[candidat] += 1
    candidat = str(input())

print("ESCRUTINI")

for i in diccandicats:
    print(f"{i} {diccandicats[i]}")

maxvots = diccandicats[max(diccandicats, key = lambda x: diccandicats[x])]

llistamax = list(filter(lambda x: diccandicats[x] == maxvots, diccandicats))

string = ""
for i in llistamax:
    string += i + " "

print(f"ELS CANDIDATS MÉS VOTATS AMB {maxvots} HAN ESTAT: {string[:-1]}")