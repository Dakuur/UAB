import pymongo
import csv

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["localitzacions"]
col = db["transactions"]

with open('comtrade.csv', 'r') as f:
    reader = csv.reader(f, delimiter=";")
    header = next(reader)
    for row in reader:
        doc = {}
        for i in range(len(header)):
            try:
                doc[header[i]] = float(row[i])
            except ValueError:
                doc[header[i]] = row[i]
        #print(doc)

        col.insert_one(doc)

print("Collection imported successfully")
