def majuscules(l):
    return list(map(lambda x: x.upper(), filter(lambda x: len(x) > 3, l)))