def palindrom(l):
    return list(filter(lambda x: x == x[::-1], l))