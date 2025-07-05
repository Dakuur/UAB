temps = []

for i in range(1, 13):
    temperatura = int(input("Introduir temperatura: "))
    temps.append(temperatura)

mitjana = sum(temps)/len(temps)

for i in range(0, len(temps)):
    temp = temps[i]
    if temp > mitjana:
        print(f"El mes {i + 1} ha tingut temperatura superior a la mitjana anual.")
    elif temp < mitjana:
        print(f"El mes {i + 1} ha tingut temperatura inferior a la mitjana anual.")
    else:
        print(f" El mes {i + 1} ha tingut temperatura igual a la mitjana anual.")