#NIF
#   nom
#   adreça
#   telèfon
#   correu
#   preferent (true o false)

def menu():
    print("1. Afegir client\n2. Eliminar client\n3. Mostra client\n4. Llistar tots els clients\n5. Llistar clients preferents\n5. Acabar")

opcio = int(input())

base = dict()

while opcio != 6:
    if opcio == 1:
        nif = input()
        while nif in base:
            print("Error: Client existent")
            nif = input()
        nom = input()
        adreça = input()
        telefon = input()
        correu = input()
        preferent = input()
        preferent = True if preferent.upper() == "S" else False
        base[nif] = {"nom": nom, "adreça": adreça, "telèfon": telefon, "correu": correu, "preferent": preferent}
    elif opcio == 2:
        nif = input()
        if nif not in base:
            print("Error: Client inexistent")
        else:
            base.pop(nif)
    elif opcio == 3:
        nif = input()
        if nif not in base:
            print("Error: Client inexistent")
        else:
            client = base[nif]
            print(f"{nif} {client['nom']} {client['adreça']} {client['telèfon']} {client['correu']} Preferent: {preferent}")
    elif opcio == 4:
        for i in base:
            print(i, base[i]["nom"])
    elif opcio == 5:
        print(list(filter(lambda x: base[x]["preferent"] == True, base)))
    else:
        print("Opció no disponible")
    opcio = int(input())

