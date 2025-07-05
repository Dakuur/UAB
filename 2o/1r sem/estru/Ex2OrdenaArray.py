def ordenaArray(v: list,indexInici: int,indexFinal: int):
    if indexInici != indexFinal and len(v) > 0:
        pos_min = posicioMinim(v, indexInici, indexFinal)
        v[indexInici], v[pos_min] = v[pos_min], v[indexInici]
        indexInici += 1
        return ordenaArray(v, indexInici, indexFinal)
    else:
        return v

def posicioMinim(v: list,indexInici: int,indexFinal: int):
   return v.index(min(v[indexInici:indexFinal + 1]))

l = [2,6,3,1,8,-10,10,11,-3,-2,50, -7000]
print(ordenaArray(l, 0, len(l) - 1))