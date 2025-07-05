base = int(input())
exp = int(input())

def modul(base,exp):
    res = 1
    if exp == 0:
        res = 1
    elif base == 0:
        res = 0
    elif exp < 0:
        exp = -exp
        base = 1/base
    else:
        for i in range(1, exp + 1):
            res = res * base

    return res

res=modul(base,exp)

print("El resultat d'elevar",base,"a la potència de", exp, "és", res)