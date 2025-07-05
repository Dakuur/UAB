#---------------------------------- PART 1 ----------------------------------#

#EXERCICI 1
class Hotel():
    def __init__(self,nom,codi_hotel,carrer,numero,codi_barri,codi_postal,telefon,latitud,longitud,estrelles):
        if type(numero) != int:
            raise TypeError ("numero ha de ser un valor enter positiu")
        if type(codi_barri) != int:
            raise TypeError("codi_barri ha de ser un valor enter positiu")
        if type(estrelles) != int:
            raise TypeError("estrelles ha de ser un valor enter positiu") 
        if type(latitud) != float:
            raise TypeError("latitud ha de ser un valor real")
        if type(longitud) != float:
            raise TypeError("longitud ha de ser un valor real")
        if 1 > estrelles or estrelles > 5:
            raise TypeError("estrelles ha de ser un valor entre 1 i 5")
        self.nom = nom
        self.codi_hotel = codi_hotel
        self.carrer = carrer
        self.numero = numero
        self.codi_barri = codi_barri
        self.codi_postal = codi_postal
        self.telefon = telefon
        self.latitud = latitud
        self.longitud = longitud
        self.estrelles = estrelles
    
    def __str__(self):
        return(str(self.nom)+" ("+str(self.codi_hotel)+") "+str(self.carrer)+", "+str(self.numero)+" "+str(self.codi_postal)+" (barri: "+str(self.codi_barri)+") "+str(self.telefon)+" ("+str(self.latitud)+","+str(self.longitud)+") "+str(self.estrelles)+" estrelles")
    
    def __gt__(self,altre_hotel):
        if self.estrelles > altre_hotel.estrelles:
            return True
        else:
            return False
        
    def distancia(self, latitud, longitud):
        raditerra = 6378.137
        long1 = self.longitud*math.pi/180
        lat1 = self.latitud*math.pi/180
        long2 = longitud*math.pi/180
        lat2 = latitud*math.pi/180
        dist = math.acos(math.sin(lat1)*math.sin(lat2)+math.cos(lat1)*math.cos(lat2)*math.cos(long2-long1))*raditerra
        return dist

#EXERCICI 2
def codi_in_llista_hotels(llista,cadena):
    trobat = False
    for i in range(0,len(llista)):
        if cadena == llista[i].codi_hotel:
            trobat = True
    return trobat

#EXERCICI 3
def importar_hotels(fitxer, separador):
    linies = []
    llistafiltrada = []
    codishotels = []
    try:
        fitxer = open(fitxer,"r", encoding="utf-8")
    except FileNotFoundError:
        print("fitxer no trobat")
    linies = fitxer.readlines()
    for i in range (1,len(linies)):
        cadena = linies[i].replace("- HB",str(separador) + "HB").replace("-  HB",str(separador) + "HB")
        llista2 = cadena.split(str(separador))
        llista2[7]=int(llista2[7])/1000000
        llista2[8]=int(llista2[8])/1000000
        if llista2[0][-1] == " ":
            llista2[0] = llista2[0][:-1]
        x = Hotel(str(llista2[0]),str(llista2[1]),str(llista2[2]),int(llista2[3]),int(llista2[4]),str(llista2[5]),str(llista2[6]),float(llista2[7]),float(llista2[8]),int(llista2[9]))
        if x.codi_hotel not in codishotels:
            llistafiltrada.append(x)
            codishotels.append(x.codi_hotel)
    print("S'han importat correctament",len(llistafiltrada),"hotels")
    return llistafiltrada
    fitxer.close()

#EXERCICI 4
class Barri():
    def __init__(self, nom, codi_districte):
        self.nom = nom
        self.codi_districte = codi_districte
        if self.codi_districte < 0 or type(self.codi_districte) != int:
            raise TypeError("codi_districte ha de ser un valor enter positiu")
    def __str__(self):
        return str(self.nom)+" (districte: "+str(self.codi_districte)+")"

#EXERCICI 5
def importar_barris(fitxer, separador):
    diccionari = {}
    try:
        fitxer = open(fitxer,"r")
    except FileNotFoundError:
        print("fitxer no trobat")
    linies = fitxer.readlines()
    for i in range (1,len(linies)):
        llista = linies[i].split(separador)
        llista[2]=llista[2][:-1]
        objecte = Barri(str(llista[2]),int(llista[1]))
        diccionari[int(llista[0])] = objecte
    print("S'han importat correctament",len(diccionari),"barris")
    return diccionari
    fitxer.close()

#EXERCICI 6
class Districte():
    def __init__(self, nom, extensio, poblacio):
        self.llista_barris = []
        self.nom = nom
        self.extensio = extensio
        self.poblacio = poblacio
        if self.poblacio < 0 or type(self.poblacio) != int:
            raise TypeError("poblacio ha de ser un valor enter positiu")
        if self.extensio < 0 or type(self.extensio) != float:
            raise TypeError("extensio ha de ser un valor real positiu")
        if len(self.llista_barris) == 0:
            self.llista_barris = ["N/D"]
    def __str__(self):
        llista = ",".join(self.llista_barris)
        return str(self.nom) + "  (" + str(self.extensio) + " kms2, " + str(self.poblacio) + " habitants)  barris: " + str(llista)
    def densitat(self):
        return self.poblacio/self.extensio

#EXERCICI 7
def importar_districtes(fitxer, separador):
    diccionari = {}
    try:
        fitxer = open(fitxer,"r")
    except FileNotFoundError:
        print("fitxer no trobat")
    linies = fitxer.readlines()
    for i in range (1,len(linies)):
        llista = linies[i].split(separador)
        llista[3]=llista[3][:-1]
        objecte = Districte(str(llista[1]),float(llista[2]),int(llista[3]))
        diccionari[int(llista[0])] = objecte
    print("S'han importat correctament",len(diccionari),"districtes")
    return diccionari
    fitxer.close()

#EXERCICI 8
def omplir_llista_barris(dicDistrictes,dicBarris):
    status = True
    for x in dicDistrictes:
        if dicDistrictes[x].llista_barris[0] != "N/D":
            status = False
    if status == True:
        for y in dicDistrictes:
            dicDistrictes[y].llista_barris.pop(0)
            for z in dicBarris:
                if dicBarris[z].codi_districte == y:
                    dicDistrictes[y].llista_barris.append(dicBarris[z].nom)
    else:
        print("El diccionari de districtes ja conté informació dels barris")

#EXERCICI 9
def mostrar_hotels(llistahotels):
    if len(llistahotels) == 0:
        print("No hi ha hotels")
        return 0
    for i in range(0, len(llistahotels)):
        print(llistahotels[i])

#---------------------------------- PART 2 ----------------------------------#

#EXERCICI 1
def ordenar_per_estrelles(llistahotels):
    return sorted(llistahotels, key=lambda hotel: hotel.estrelles)

#EXERCICI 2
def mostrar_noms_hotels(llistahotels):
    if len(llistahotels) == 0:
        print("No hi ha hotels")
        return 0
    for i in range(0, len(llistahotels)):
        print(llistahotels[i].nom + " (" + llistahotels[i].codi_hotel + ")")

#EXERCICI 3
def buscar_per_nom(llistahotels, cadena):
    resultats = []
    for i in range(0, len(llistahotels)):
        if cadena.lower() in llistahotels[i].nom.lower():
            resultats.append(llistahotels[i])
    return resultats

#EXERCICI 4
def buscar_per_estrelles(llistahotels, estrella):
    return list(filter(lambda n: n.estrelles == estrella, llistahotels))

#EXERCICI 5
def buscar_hotels(llistahotels):
    mode = int(input("Introdueix criteri de cerca (1 - per nom, 2 - per estrelles): "))
    if mode == 1:
        nom = input("Nom de l'hotel: ")
        resultats = buscar_per_nom(llistahotels, nom)
        if len(resultats) == 0:
            print("No s'han trobat hotels")
        else:
            print("S'han trobat " + str(len(resultats)) + " hotels amb aquest nom")
            mostrar_noms_hotels(resultats)
    elif mode == 2:
        correcte = False
        while correcte == False:
            estrelles = input("Número d'estrelles: ")
            try:
                estrelles = int(estrelles)
            except:
                raise ValueError("Error: el número d'estrelles ha de ser un valor enter")
            if int(estrelles) not in range(1, 6):
                print("Error: el número d'estrelles ha de ser un valor entre 1 i 5")
            else:
                correcte = True
        buscat = buscar_per_estrelles(llistahotels, estrelles)
        if len(buscat) == 0:
            print("No s'han trobat hotels")
        else:
            print("S'han trobat", str(len(buscat)),"hotels de", str(estrelles),"estrelles")
            mostrar_noms_hotels(buscat)
    else:
        print("Error: criteri de cerca no vàlid")

#EXERCICI 6
def hotel_mes_proper(llistahotels, latitud, longitud):
    millor = llistahotels[0]
    for i in range(0, len(llistahotels)):
        if llistahotels[i].distancia(latitud, longitud) < millor.distancia(latitud, longitud):
            millor = llistahotels[i]
    return millor, millor.distancia(latitud, longitud)

#---------------------------------- PART 3 ----------------------------------#

#EXERCICI 1
def ordenar_per_nom(llistahotels):
    return sorted(llistahotels, key=lambda hotel: hotel.nom.lower())

#EXERCICI 2
def carrers_amb_hotels(llistahotels):
    llistacarrers = []
    for i in range(0, len(llistahotels)):
        llistacarrers.append(llistahotels[i].carrer)
    return list(set(llistacarrers))

#EXERCICI 3
def estrelles_per_barri(llistahotels,dicBarris):
    diccionari = {}
    diccionari2 = {}
    for x in dicBarris:
        llista = [0,0,0,0,0]
        diccionari2[x] = llista
    for x in llistahotels:
            valor = x.estrelles
            diccionari2[x.codi_barri][int(valor)-1]=diccionari2[x.codi_barri][int(valor)-1]+1
    for x in dicBarris:
            diccionari[dicBarris[x].nom]=diccionari2[x]
    return diccionari

#EXERCICI 4
def densitat_per_districte(llistahotels,dicBarris,dicDistrictes):
    diccionari = {}
    llista = []
    for x in dicDistrictes:
        for y in dicDistrictes[x].llista_barris:
            for z in llistahotels:
                if str(dicBarris[z.codi_barri].nom) == str(y):
                    llista.append(z)
        total = len(llista)
        densitat=total/dicDistrictes[x].extensio
        diccionari[x]=densitat
    return diccionari

#EXERCICI 5
def afegir_prefixe_int(hotel):
    if hotel.telefon[0] != "+":
        hotel.telefon = "+34" + str(hotel.telefon)
    return hotel

#EXERCICI 6
def modificar_telefons(llistahotels):
    llistahotels = list(map(afegir_prefixe_int, llistahotels))
    return llistahotels

#EXERCICI 7
def mostrar_menu():
    print("\n--- MENÚ PRINCIPAL ---\n1 - Veure hotels\n2 - Veure hotels per estrelles\n3 - Buscar hotels\n4 - Buscar hotel proper\n5 - Llistat alfabètic d'hotels\n6 - Carrers amb hotels\n7 - Estadística per barris\n8 - Estadística per districtes\n9 - Internacionalitzar telèfons\nS - Sortir del programa\n")

#EXERCICI 8
import math

try:
    llistahotels = importar_hotels("hotels.csv",";")
    Dicbarris = importar_barris("barris.csv",";")
    Dicdistrictes = importar_districtes("districtes.csv",";")
except FileNotFoundError:
    print("Error llegint fitxers: ", FileNotFoundError)
except Exception as message:
    if Exception != FileNotFoundError:
        print("Error processant els fitxers: ",message)
else:
    omplir_llista_barris(Dicdistrictes,Dicbarris)
    opcio = ""
    while opcio.lower() != "s":
        mostrar_menu()
        opcio = str(input("Introdueix una de les opcions del menú: "))
        if opcio == "1":
            mostrar_hotels(llistahotels)
        elif opcio == "2":
            llistaordenada = ordenar_per_estrelles(llistahotels)
            mostrar_hotels(llistaordenada)
        elif opcio == "3":
            buscar_hotels(llistahotels)
        elif opcio == "4":
            try:
                latitud = float(input("Introdueix latitud: "))
                longitud = float(input("Introdueix longitud: "))
            except ValueError:
                print("Error: latitud i longitud han de ser valors reals")
            else:
                hotel, distancia = hotel_mes_proper(llistahotels, latitud, longitud)
                print("L'hotel més proper és el",hotel.nom,"a",distancia,"kms")
        elif opcio == "5":
            llistaordenada = ordenar_per_nom(llistahotels)
            mostrar_hotels(llistaordenada)
        elif opcio == "6":
            llistacarrers = carrers_amb_hotels(llistahotels)
            print("Hi ha",len(llistacarrers),"carrers amb algun hotel:")
            for i in range(0, len(llistacarrers)):
                print(llistacarrers[i])
        elif opcio == "7":
            diccionari = estrelles_per_barri(llistahotels,Dicbarris)
            for x in diccionari:
                print(str(x)+": "+str(diccionari[x][0])+" hotels de 1 estrella, "+str(diccionari[x][1])+" hotels de 2 estrelles, "+str(diccionari[x][2])+" hotels de 3 estrelles, "+str(diccionari[x][3])+" hotels de 4 estrelles, "+str(diccionari[x][4])+" hotels de 5 estrelles")
        elif opcio == "8":
            res = densitat_per_districte(llistahotels, Dicbarris, Dicdistrictes)
            for x in res:
                print("Districte", str(x) + ":", round(res[x], 2), "hotels/km2")
        elif opcio == "9":
            modificar_telefons(llistahotels)
        elif opcio.lower() == "s":
            print("Sortint del programa")
        else:
            print("Opció no permesa")
finally:
    print("© David Morillo Massagué (NIU: 1666540) - Adrià Muro Gómez (NIU: 1665191)")