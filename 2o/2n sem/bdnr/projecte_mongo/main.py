import argparse
import json
from pymongo import MongoClient

def import_data(filename, db):
    with open(filename, 'r', encoding="utf-8") as file:
        data = json.load(file)
    
    for key, value in data.items():
        collection = db[key]

        delete_result = collection.delete_many({})
        print(f"{delete_result.deleted_count} documents deleted from collection '{key}'.")

        result = collection.insert_many(value)
        print(f"{len(result.inserted_ids)} documents inserted into collection '{key}'.")

def main():
    parser = argparse.ArgumentParser(description='Import JSON data into MongoDB.')
    parser.add_argument('-f', '--filename', type=str, required=True, help='JSON file to import')
    
    args = parser.parse_args()
    
    client = MongoClient('dcccluster.uab.es', 8214)
    db = client.projecte
    
    import_data(args.filename, db)

if __name__ == "__main__":
    main()

# per executar, fer servir:
# py main.py -f data.json