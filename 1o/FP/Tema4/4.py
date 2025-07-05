entrada = str(input())
contador = 0
vocals = ["a", "e", "i", "o", "u", "A", "E", "I", "O", "U"]

for i in range(0, len(entrada)):
    if entrada[i] in vocals:
        contador = contador + 1

print('El string "'+entrada+'" té',contador,'vocals.')