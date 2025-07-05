def cicle(x):
    if not isinstance(x, (list, str, tuple)):
        raise TypeError("Tipus Incorrecte")
    while True:
        for i in x:
            yield i