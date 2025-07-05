nombre = int(input())

if nombre < 0:
    print("ERROR")
elif nombre > 200:
    print("ERROR")
else:
    ascii = chr(nombre)
    ascii2 = chr(nombre+1)
    ascii3 = chr(nombre+10)

    print(ascii,ascii2,ascii3)