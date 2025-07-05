def majuscules(l):
    filtre = list(map(lambda x: x.upper(), filter(lambda x: len(x) > 3 and x[-1] in "aeiou", l)))
    return [list(s) for s in filtre]

llista = ['pere', 'joan', 'marti', 'pol', 'aina', 'oriol', 'marta', 'ona', 'silvia', 'pau', 'josep']

print(majuscules(llista))