from copy import copy

def descRec(l: list, i: int):
    if l[i] > 1:
        l[i] -= 1
        if i == -1:
            l.append(1)
        else:
            l[i+1] += 1
    else:
        descRec(l, i - 1)
    return l

def desc(n: int):
    descomposicions = []
    res = descRec([n], -1)
    descomposicions.append(res)
    while res != n*[1]:
        res = descRec(res.copy(), -1)
        if res == sorted(res, reverse=True):
            descomposicions.append(res)
    return descomposicions


if __name__ == "__main__":

    num = int(input("Número a descomposar:"))

    resultat = desc(num)

    print(f"Descomposició per a N = {num}:")

    for i in resultat:
        print(i)