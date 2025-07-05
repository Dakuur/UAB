#print(input().upper())

entrada = str(input())

sortida = ""

for i in range(0, len(entrada)):
    ascii = ord(entrada[i])
    if ascii in range(97, 123):
        ascii = ascii - 32
    j = chr(ascii)
    sortida = sortida + str(j)

print(sortida)