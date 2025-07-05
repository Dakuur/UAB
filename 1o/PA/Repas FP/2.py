def factorial(numero):
    if numero == 1:
        return 1
    else:
        return numero * factorial(numero - 1)

def factorial_llista(llista):
    res = list()
    for element in llista:
        res.append(factorial(element))
    return res