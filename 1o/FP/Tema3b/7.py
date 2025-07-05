print("Menú\n\n1 - Opció 1\n2 - Opció 2\n3 - Opció 3\n\nPrem una tecla per seleccionar opció:")

num = int(input())

if num in [1,2,3]:
    print("Fent tasca", str(num))
else:
    print("Opció incorrecta")


