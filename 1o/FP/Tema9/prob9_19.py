def suspesos(l):
    return list(map(lambda x: x[0], filter(lambda x: x[1] < 5, l)))