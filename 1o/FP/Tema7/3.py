cadena = str(input())
operand = str(input())

if operand not in "&#*":
    raise SyntaxError("ERROR: Operació no definida")

def operacions_strings(cadena, operador):
    if operador == '&':
        try:
            return float(cadena)
        except ValueError:
            raise ValueError("ERROR: La cadena no conté un número")
    elif operador == '#':
        try:
            index = int(input("Introdueix la posició del caràcter a extreure: "))
            return cadena[index]
        except ValueError:
            raise ValueError("ERROR: Valor no enter en l’índex")
        except IndexError:
            raise IndexError("ERROR: Intent d'accés a fora de la cadena")
    elif operador == '*':
        try:
            repet = int(input("Introdueix el nombre de repeticions: "))
            return cadena*repet
        except ValueError:
            raise ValueError("ERROR: Valor no enter en el nombre de repeticions")
    else:
        raise SyntaxError("Operació no definida")

res = operacions_strings(cadena, operand)
print("Resultat de l'operació:", str(res))