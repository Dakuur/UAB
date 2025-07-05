import networkx as nx

G = nx.read_graphml("G_BFS.graphml")
print(G)

G = nx.read_graphml("G_DFS.graphml")
print(G)

import pandas as pd

def contar_valors_unics(arxiu_csv, columna):
    df = pd.read_csv(arxiu_csv)

    valors_unics = df[columna].nunique()

    return valors_unics

arxiu_csv = 'TrackData.csv'
columna = 'track_id'

resultat = contar_valors_unics(arxiu_csv, columna)
print(f'Hi ha {resultat} valors únics a la columna "{columna}" de l\'arxiu CSV.')

columna = 'artist_id'

resultat = contar_valors_unics(arxiu_csv, columna)
print(f'Hi ha {resultat} valors únics a la columna "{columna}" de l\'arxiu CSV.')

columna = 'album_id'

resultat = contar_valors_unics(arxiu_csv, columna)
print(f'Hi ha {resultat} valors únics a la columna "{columna}" de l\'arxiu CSV.')