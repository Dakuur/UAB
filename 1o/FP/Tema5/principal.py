from llistes import LlegirLlista, InicialitzarLlista, MaximLlista, MinimLlistaNoZero

temps = LlegirLlista(14)

histograma = InicialitzarLlista(61, 0)

for i in range(0, len(temps)):
    histograma[temps[i]+10] += 1

max=MaximLlista(histograma)-10
min=MinimLlistaNoZero(histograma)-10

print("Temperatura més repetida:", max)
print("Temperatura menys repetida:", min)