def mergeSortArray(v, indexInici, indexFinal):
    if indexInici < indexFinal:
        mid = (indexInici + indexFinal) // 2

        mergeSortArray(v, indexInici, mid)
        mergeSortArray(v, mid + 1, indexFinal)

        merge(v, indexInici, mid, indexFinal)

def merge(v, indexInici, mid, indexFinal):
    esquerra = v[indexInici:mid + 1]
    dreta = v[mid + 1:indexFinal + 1]

    i = j = 0
    x = indexInici # Indexador array

    while i < len(esquerra) and j < len(dreta):
        if esquerra[i] <= dreta[j]:
            v[x] = esquerra[i]
            i += 1
        else:
            v[x] = dreta[j]
            j += 1
        x += 1

    while i < len(esquerra):
        v[x] = esquerra[i]
        i += 1
        x += 1

    while j < len(dreta):
        v[x] = dreta[j]
        j += 1
        x += 1