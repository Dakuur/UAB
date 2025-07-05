import math

nombre = int(input("Nombre: "))

milers = str(math.trunc(nombre/1000))

ultim = milers[-1]

print("Les unitats de miler del nombre", str(nombre), "són", str(ultim))