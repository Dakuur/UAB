def frec(vec: list, n) -> int:
    if n==0:
        return vec[0]
    else:
        return vec[n]*frec(vec, n-1)
    
def fiter(vec: list, n) -> int:
    pila = []
    while n != 0:
        pila.append(vec[n])
        n -= 1

    res = 1
    while pila:
        res *= pila.pop()
    return res

v = [1,2,3,4,3,4,2,52,2,5,3]
print(frec(v, len(v) - 1))