from math import sqrt

def equacio(a, b, c):
    delta = b**2 - 4*a*c
    if delta < 0:
        return 0, None, None
    elif delta == 0:
        x1 = -b / (2*a)
        return 1, x1, None
    else:
        x1 = (-b + sqrt(delta)) / (2*a)
        x2 = (-b - sqrt(delta)) / (2*a)
        return 2, x1, x2
'''
a = int(input("Introdueix el coeficient a: "))
b = int(input("Introdueix el coeficient b: "))
c = int(input("Introdueix el coeficient c: "))

num_arrels, x1, x2 = equacio(a, b, c)
if num_arrels == 0:
    print("L'equació no té solucions reals")
elif num_arrels == 1:
    print("L'equació té una arrel real:", x1)
else:
    print("L'equació té dues arrels reals:", str(x1) + ", " + str(x2))
'''