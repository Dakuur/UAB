import json
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.localitzacions

collection = db.espanya

json_file = 'geonames_ES.json'

with open(json_file, encoding="utf-8") as file:
    data = json.load(file)
    for item in data:

        item['_id'] = item['geonameid']
 
        item['localització'] = {
            'type': 'Point',
            'coordinates': [item['latitude'], item['longitude']]
        }

        del item['geonameid']
        del item['latitude']
        del item['longitude']
   
    collection.insert_many(data)

print("Dades importades correctament a la base de dades MongoDB amb les restriccions aplicades.")
