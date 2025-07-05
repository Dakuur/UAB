import networkx as nx
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import csv

#CONSTANTS

CLIENT_ID = "4d4b0e7e841e47e3a24f4999b8a3997d"
CLIENT_SECRET = "df7ce55f471f41db8f57d50c2896dd30"
            
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
    results = sp.search(q='artist:' + artist_name, type='artist')
        
    if len(results['artists']['items']) > 0:
        artist_id = results['artists']['items'][0]['id']
        
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
    graph = nx.DiGraph()
    visited = set()
    queue = [search_artist(sp, seed)]
    
    while len(queue) > 0 and len(visited) < max_nodes_to_crawl:
        if strategy == "BFS":
            act_artist = queue.pop(0)
            
        else:
            act_artist = queue.pop()
 
        if act_artist not in visited:
            visited.add(act_artist)
        
            try:
                print("Getting artist data\n")
                related_artists = sp.artist_related_artists(act_artist)['artists']
            except:
                break
        
            for related_artist in related_artists:
                related_id = related_artist['id']
                if related_id not in visited:
                    queue.append(related_id)
            
                graph.add_edge(act_artist, related_id)
    dict_explored = {node: True if node in visited else False for node in graph.nodes()}
    nx.set_node_attributes(graph, dict_explored, name="Explored")
    
    nx.write_graphml(graph, out_filename)
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
        dict_explored = nx.get_node_attributes(graph, "Explored")
        artists.update([node for node in graph.nodes() if dict_explored[node]])
    print(f"Number of different artists: {len(artists)}")
    print(f"Getting songs data for {len(artists)} artists. Aprox.: {len(artists) * 10} songs")

    header = [
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
    track_data = [header]
    
    track_ids = dict()

    for artist in list(artists):
        print("Getting artist data\n")
        top_tracks = sp.artist_top_tracks(artist, country="ES")["tracks"]
        track_ids.update({x["id"]: x for x in top_tracks})

    print("Getting track data\n")
    track_features_list = sp.audio_features(list(track_ids.keys()))
    track_keys = list(track_ids.keys())
    for i in range(0, len(track_keys)):
        key = track_keys[i]
        track_ids[key]["features"] = track_features_list[i]

    for track in list(track_ids.values()):
        album = track['album']
        print("Adding new row\n")
        song_info = [
            track['id'],
            track['duration_ms'],
            track['name'],
            track['popularity'],
            track["features"]['danceability'],
            track["features"]['energy'],
            track["features"]['loudness'],
            track["features"]['speechiness'],
            track["features"]['acousticness'],
            track["features"]['instrumentalness'],
            track["features"]['liveness'],
            track["features"]['valence'],
            track["features"]['tempo'],
            album['id'],
            album['name'],
            album['release_date'],
            artist,
            sp.artist(artist)['name']
        ]
        track_data.append(song_info)

    with open(out_filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(track_data)

    print("Data obtained successfully!")

    print(f"Number of tracks in the datset: {len(track_data)}")
    print(f"Number of different artists: {len(artists)}")

    return None


if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    #CREACIÓ OBJECTE MAIN

    auth_manager = SpotifyClientCredentials(client_id = CLIENT_ID, client_secret = CLIENT_SECRET)
    sp = spotipy.Spotify (auth_manager = auth_manager)
    graphs = []
    try:
        G_bfs = nx.read_graphml("G_BFS.graphml")
    except:
        G_bfs = crawler(sp, "Taylor Swift", 100, strategy = "BFS", out_filename = "G_BFS.graphml")
    graphs.append(G_bfs)
    print("Completed", G_bfs)
    try:
        G_dfs = nx.read_graphml("G_DFS.graphml")
    except:
        G_dfs = crawler(sp, "Taylor Swift", 100, strategy = "DFS", out_filename = "G_DFS.graphml")
    graphs.append(G_dfs)
    print("Completed", G_dfs)
    get_track_data(sp, graphs, "TrackData.csv")

    # ------------------- END OF MAIN ------------------------ #