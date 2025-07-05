from pymongo import MongoClient
import json
from options import Options
import sys
"""
FET PER: LUCIA GARRIDO, DAVID MORILLO, ADRIÀ MURO, ALBERT GUILLAUMET
"""


Host = 'localhost' # localhost per connexions a la màquina main
Port = 27017

DSN = "mongodb://{}:{}".format(Host,Port)
conn = MongoClient(DSN)

#XARXA SOCIAL

bd = conn['Entrega5']

#Exercici 5. Insertar dades d'un document Json a la base de dades des de Python.
with open('Entrega5.Habitatges.json', 'r') as file:
    data_json = json.load(file)

col = bd['habitat']
col.drop() #Afegim aio per esborrar i inserir cada vegada que executem, sino ens dona error.
col.insert_many(data_json)


#Exercici 6.

print(bd.list_collection_names())

if "Habitatges" not in bd.list_collection_names():
    print("Els Habitatges no estan carregats. Sortint")
    sys.exit(0)

Habitatges = bd["habitat"]
poblacio = Habitatges.find({"adreca.poblacio": 'Sant Cugat'})
for p in poblacio:
    print("\nHabitatges:  ",p)
habitatge = Habitatges.find({"adreca.numero":70})
for h in habitatge:
    print("\nHabitatge qualsevol:  ",h)




#DETECCIÓ DE CÀNCER
bd = conn['investigacio']

#Exercici 8. Insertar dades d'un document Json a la base de dades des de Python.
with open('investigacio.casos.json', 'r') as file:
    data_json = json.load(file)

col = bd['casos']
col.drop() # Añadimos esta línea para borrar e insertar cada vez que ejecutamos, si no nos da error.
col.insert_many(data_json)


#Exercici 9.
if "casos" not in bd.list_collection_names():
    print("Els casos no estan carregats. Sortint")
    sys.exit(0)

casos = bd["casos"]
casos_inflamatori = casos.find({"diagnostic_final": "inflamatori"})
casos_carcinoma_lobular = casos.find({"diagnostic_final": "carcinoma lobular"})

print("\nCasos amb diagnòstic final 'inflamatori':")
for cas in casos_inflamatori:
    print(cas)

print("\nCasos amb diagnòstic final 'carcinoma lobular':")
for caso in casos_carcinoma_lobular:
    print(caso)

cas_id_especifica = casos.find_one({"ID_cas": "3"})
print("\nCas amb ID específica:")
print(cas_id_especifica)

conn.close()


