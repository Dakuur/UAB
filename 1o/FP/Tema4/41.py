contrasenya = str(input())

bien = True

#longitud
if len(contrasenya) < 8:
    bien = False

#minuscules, majuscules
majuscula = False
minuscula = False
for i in range(0, len(contrasenya)):
    if contrasenya[i] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        majuscula = True
        break
for i in range(0, len(contrasenya)):
    if contrasenya[i] in "abcdefghijklmnopqrstuvwxyz":
        minuscula = True
        break
if minuscula == False or majuscula == False:
    bien = False

#mes de 2 digits
digits = 0
for i in range(0, len(contrasenya)):
    if digits >= 2:
        break
    if contrasenya[i] in "1234567890":
        digits = digits + 1
if digits < 2:
    bien = False

#caracters especials
especial = False
for i in range(0, len(contrasenya)):
    if contrasenya[i] in "$&*€_@#":
        especial = True
        break
if especial == False:
    bien = False

if bien == False:
    print("Error: Contrasenya no vàlida.")
else:
    comprovacio=str(input())
    if comprovacio != contrasenya:
        print("Error: Les dues contrasenyes no són iguals.")
    else:
        print("Contrasenya correcta.")