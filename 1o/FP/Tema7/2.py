cadena = str(input())
operand = str(input())

if operand not in "&#*":
    raise SyntaxError("ERROR: Operació no definida")

def funcio(cadena, operand):
    if operand == "&":
        try:
            resultat = float(cadena)
        except:
            raise ValueError("ERROR: La cadena no conté un número")
    elif operand == "#":
        try:
            posicio = int(input())
        except:
            raise ValueError("ERROR: Valor no enter en l’índex")
        try:
            resultat = cadena[posicio]
        except:
            raise ValueError("ERROR: Intent d'accés a fora de la cadena")
    elif operand == "*":
        try:
            num = int(input())
        except:
            raise ValueError("ERROR: Valor no enter en el nombre de repeticions")
        resultat = str(cadena) * num
    
    print("Resultat de l'operació:", str(resultat))

funcio(cadena, operand)