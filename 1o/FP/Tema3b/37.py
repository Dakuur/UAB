#print("MENÚ\n1.- Suma\n2.- Resta\n3.- Producte\n4.- Divisió\n5.- Sortir\nSelecciona una de les opcions:")
operacio = int(input())

operacions = {
    1 : "+",
    2 : "-",
    3 : "x",
    4 : "/",
}

if operacio < 1 or operacio > 5:
    print("Error: Opció no vàlida")
elif operacio == 5:
        print("Sortint de la calculadora...")

else:

    num1 = float(input())
    num2 = float(input())

    if operacio == 4 and num2 == 0:
        print("Error: Divisió per zero")
    else:
        if operacio == 1:
            resultat = num1 + num2
        elif operacio == 2:
            resultat = num1 - num2
        elif operacio == 3:
            resultat = num1 * num2
        elif operacio == 4:
            resultat = num1 / num2

        print(num1, operacions[operacio], num2, "=", resultat)