import json

def load_json_file(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        data = json.load(file)
    return data

clients_data = load_json_file("database_files/clients.json")
cotxes_data = load_json_file("database_files/cotxes.json")
estades_data = load_json_file("database_files/estades.json")
places_parking_data = load_json_file("database_files/places_parking.json")
productes_data = load_json_file("database_files/productes.json")
tickets_data = load_json_file("database_files/tickets.json")

all_data = {
    "clients": clients_data,
    "cotxes": cotxes_data,
    "estades": estades_data,
    "places_parking": places_parking_data,
    "productes": productes_data,
    "tickets": tickets_data
}

with open("data.json", "w", encoding='utf-8') as outfile:
    json.dump(all_data, outfile, indent=2, ensure_ascii=False)

print("Combined data saved to 'data.json'")
