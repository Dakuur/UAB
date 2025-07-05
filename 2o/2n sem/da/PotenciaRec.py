def PotenciaRec(b, n):
    if n == 0:
        return 1
    if n%2 == 0:
        pot = PotenciaRec(b, n/2)
        return pot*pot
    else:
        pot = PotenciaRec(b, (n-1)/2)
        return pot*pot*b