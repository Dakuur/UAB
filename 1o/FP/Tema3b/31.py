eur = float(input())
divisa = int(input())
moneda = {1 : 1.34, 2 : 0.83, 3 : 1.23, 4 : 133.11 }
moneda2 = {1 : "USD", 2 : "GBP", 3 : "CHF", 4 : "JPY" }
if divisa not in moneda:
    print("Error: moneda no disponible")
else:
    print(eur, "euros són", str(eur*moneda[divisa]), moneda2[divisa])