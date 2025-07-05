import networkx as nx
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import csv

"""
David Morillo Massagué (1666540)
Adrià Muro Gómez (1665191)
"""

#CONSTANTS

CLIENT_ID = "478a456312474b76a1c69f8eb08b7df0"
CLIENT_SECRET = "c26176811ace4da2a509e8d79b4f9a15"

            
# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #


# --------------- END OF AUXILIARY FUNCTIONS ------------------ #


def search_artist(sp: spotipy.client.Spotify, artist_name: str) -> str:
    """
    Search for an artist in Spotify.

    :param sp: spotipy client object
    :param artist_name: name to search for.
    :return: spotify artist id.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    results = sp.search(q='artist:' + artist_name, type='artist') #Busca relacions amb el nom donat
        
    if len(results['artists']['items']) > 0:
        artist_id = results['artists']['items'][0]['id'] #retorna la id de l'artista
        
    else:
        return None
    
    # ----------------- END OF FUNCTION --------------------- #
    
    return artist_id

def crawler(sp: spotipy.client.Spotify, seed: str, max_nodes_to_crawl: int, strategy: str = "BFS",
            out_filename: str = "g.graphml") -> nx.DiGraph:
    """
    Crawl the Spotify artist graph, following related artists.

    :param sp: spotipy client object
    :param seed: starting artist id.
    :param max_nodes_to_crawl: maximum number of nodes to crawl.
    :param strategy: BFS or DFS.
    :param out_filename: name of the graphml output file.
    :return: networkx directed graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    
    #Inicialització de variables
    
    graph = nx.DiGraph()
    visited = set()
    seed_id = search_artist(sp, seed)
    queue = [seed_id]

    info_seed = sp.artist(seed_id)

    graph.add_node(seed_id)
    graph.nodes[seed_id]["name"] = info_seed["name"]
    graph.nodes[seed_id]["id"] = seed_id
    graph.nodes[seed_id]["followers"] = info_seed["followers"]["total"]
    graph.nodes[seed_id]["popularity"] = info_seed["popularity"]
    graph.nodes[seed_id]["genres"] = str(info_seed["genres"])
    
    while len(queue) > 0 and len(visited) < max_nodes_to_crawl: #Mentres la queue no estigui buida i no es superi el límit de nodes a visitar
        if strategy == "BFS": #BFS
            act_artist = queue.pop(0)
            
        else: #DFS
            act_artist = queue.pop()
 
        if act_artist not in visited:
            visited.add(act_artist)
        
            try:
                related_artists = sp.artist_related_artists(act_artist)['artists'] #Busca els 20 artistes més relacionats
            except:
                break
        
            for related_artist in related_artists:
                related_id = related_artist['id']

                # info de artista en graf
                graph.add_node(related_id)
                graph.nodes[related_id]["name"] = related_artist["name"]
                graph.nodes[related_id]["id"] = related_id
                graph.nodes[related_id]["followers"] = related_artist["followers"]["total"]
                graph.nodes[related_id]["popularity"] = related_artist["popularity"]
                graph.nodes[related_id]["genres"] = str(related_artist["genres"])

                if related_id not in visited:
                    queue.append(related_id)
            
                graph.add_edge(act_artist, related_id)
                
    dict_explored = {node: True if node in visited else False for node in graph.nodes()}
    nx.set_node_attributes(graph, dict_explored, name="explored") #Per evitar iteracions de més en get_track_data
    
    nx.write_graphml_xml(graph, out_filename)
    return graph

    # ----------------- END OF FUNCTION --------------------- #

def get_track_data(sp: spotipy.client.Spotify, graphs: list, out_filename: str) -> pd.DataFrame:
    """
    Get track data for each visited artist in the graph.

    :param sp: spotipy client object
    :param graphs: a list of graphs with artists as nodes.
    :param out_filename: name of the csv output file.
    :return: pandas dataframe with track data.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    
    artists = set()
    for graph in graphs:
        dict_explored = nx.get_node_attributes(graph, "explored")
        artists.update([node for node in graph.nodes() if dict_explored[node]]) #Només itera pels que s'han explorat

    print(f"Getting songs data for {len(artists)} artists. Aprox.: {len(artists) * 10} songs")
    added = []
    header = [ #Header
        'track_id', 
        'duration', 
        'name', 
        'popularity', 
        'danceability', 
        'energy', 
        'loudness', 
        'speechiness', 
        'acousticness', 
        'instrumentalness', 
        'liveness', 
        'valence', 
        'tempo', 
        'album_id', 
        'album_name', 
        'album_release_date', 
        'artist_id', 
        'artist_name'
        ]
    track_data = []
    track_data.append(header)
    
    for artist in list(artists):
        top_tracks = sp.artist_top_tracks(artist, country="ES")
        for track in top_tracks["tracks"]:
            if track["id"] in added: # Per a evitar repetits
                continue
            print("Getting track data\n")
            track_features = sp.audio_features(track['id'])[0] #track features (música)
            album = track['album'] #album
            song_info = [
                track['id'],
                track['duration_ms'],
                track['name'],
                track['popularity'],
                track_features['danceability'],
                track_features['energy'],
                track_features['loudness'],
                track_features['speechiness'],
                track_features['acousticness'],
                track_features['instrumentalness'],
                track_features['liveness'],
                track_features['valence'],
                track_features['tempo'],
                album['id'],
                album['name'],
                album['release_date'],
                artist,
                sp.artist(artist)['name']
            ]
            track_data.append(song_info)
            added.append(track['id'])

    with open(out_filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(track_data) #Convertir a CSV

    print("Data obtained successfully!")
    #return pd.DataFrame(track_data[1:], columns = track_data[0])


if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    #CREACIÓ OBJECTE MAIN
    auth_manager = SpotifyClientCredentials(client_id = CLIENT_ID, client_secret = CLIENT_SECRET)
    sp = spotipy.Spotify (auth_manager = auth_manager)
    graphs = []
    G_bfs = crawler(sp, "Taylor Swift", 100, strategy = "BFS", out_filename = "gB.graphml")
    graphs.append(G_bfs)
    G_dfs = crawler(sp, "Taylor Swift", 100, strategy = "DFS", out_filename = "gD.graphml")
    graphs.append(G_dfs)
    get_track_data(sp, graphs, "TrackData.csv")

    # ------------------- END OF MAIN ------------------------ #