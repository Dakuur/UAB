def random_bdays(n):
    import random
    llista = []
    for i in range(0, n):
        num = random.randint(1, 366)
        llista.append(num)
    return llista

def has_duplicates(llista):
    return len(llista) != len(set(llista))

def count_matches(numpersones, numexperiments):
    coincidencies = 0
    for i in range(0, numexperiments):
        llista = random_bdays(numpersones)
        if has_duplicates(llista):
            coincidencies += 1
    return coincidencies