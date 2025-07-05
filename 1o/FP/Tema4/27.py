mesos = ["gener", "febrer", "març", "abril", "maig", "juny", "juliol", "agost", "setembre", "octubre", "novembre", "desembre"]

temps = []
suma = 0

for i in range(0, 12):
    temps.append(int(input()))
    suma = suma + temps[i]

mitjana = suma/12

print("Mes amb temperatura mínima:", temps.index(min(temps))+1)
print("Mes amb temperatura màxima:", temps.index(max(temps))+1)
print("Temperatura mitjana:", mitjana)


for i in  range(0, len(temps)):
    if temps[i] < mitjana:
        print("El mes", i+1, "ha tingut temperatura inferior a la mitjana anual.")
    elif temps[i] > mitjana:
        print("El mes", i+1, "ha tingut temperatura superior a la mitjana anual.")
    else:
        print("El mes", i+1, "ha tingut temperatura igual a la mitjana anual.")