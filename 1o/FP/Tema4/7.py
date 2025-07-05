string = str(input())

inici = int(input())
final = int(input())

while inici < 0 or final >= len(string):
    inici = int(input())
    final = int(input())

i = inici

#els caràcters de la cadena corresponents als índexs que estan entre el primer valor i el segon valor (ambdós inclosos).
print(string[inici:final+1])

#els caràcters de la cadena corresponents als índexs parells que estan entre el primer valor i el segon valor (ambdós inclosos).
llista = ""
if inici%2 != 0:
    inici = inici + 1
for n in range(inici, final + 1, 2):
    if n <= len(string):
        llista = llista + string[n]

inici = i
print(llista)

#els caràcters de la cadena que van des del primer caràcter de la cadena fins el caràcter que està en la posició indicada pel primer valor (aquest inclòs).
print(string[:inici+1])

#els caràcters de la cadena que van des del caràcter que està en la posició indicada pel segon valor (aquest inclòs) fins el final de la cadena.
print(string[final:])