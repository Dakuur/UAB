def sumaXifres(n: str):
    n = str(n)
    if n[0] == "-":
        n = n[1:]
    if len(n) > 1:
        return int(n[0]) + sumaXifres(n[1:])
    else:
        return int(n)
    
print(sumaXifres(-4543))