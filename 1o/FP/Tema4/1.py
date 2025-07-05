missatge = str(input()) + " "
for cha in range(0, len(missatge)):
    if missatge[cha] == "#":
        inici = cha
        while missatge[cha] != " " and cha < len(missatge):
            cha = cha + 1
        final = cha
        print(missatge[inici:final])