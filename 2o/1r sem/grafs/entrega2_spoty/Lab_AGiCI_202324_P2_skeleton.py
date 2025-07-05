import networkx as nx
import pandas as pd
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #

# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def retrieve_bidirectional_edges(g: nx.DiGraph, out_filename: str) -> nx.Graph:
    """
    Convert a directed graph into an undirected graph by considering bidirectional edges only.

    :param g: a networkx digraph.
    :param out_filename: name of the file that will be saved.
    :return: a networkx undirected graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    G = nx.Graph()
    for (u,v) in g.edges():
        if u in g[v]:
            G.add_edge(u, v)
            
    nx.write_graphml_xml(G, out_filename)
    return G
    # ----------------- END OF FUNCTION --------------------- #


def prune_low_degree_nodes(G: nx.Graph, min_degree: int, out_filename: str) -> nx.Graph:
    """
    Prune a graph by removing nodes with degree < min_degree.

    :param g: a networkx graph.
    :param min_degree: lower bound value for the degree.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    # 1rst phase
    g = G.copy()
    nodes_to_remove = []
    for node_degree in g.degree():
        if node_degree[1] < min_degree:
            nodes_to_remove.append(node_degree[0])
            
    for node in nodes_to_remove:
        g.remove(node)
    
    
    # 2nd phase
    to_clean = []
    for node_degree in g.degree():
        if node_degree[1] == 0:
            to_clean.append(node_degree[0])
            
    for node in to_clean:
        g.remove(node)
    
    
    nx.write_graphml_xml(g, out_filename)
    return g
    
    # ----------------- END OF FUNCTION --------------------- #


def prune_low_weight_edges(g: nx.Graph, min_weight=None, min_percentile=None, out_filename: str = None) -> nx.Graph:
    """
    Prune a graph by removing edges with weight < threshold. Threshold can be specified as a value or as a percentile.

    :param g: a weighted networkx graph.
    :param min_weight: lower bound value for the weight.
    :param min_percentile: lower bound percentile for the weight.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    if min_weight is not None and min_percentile is not None:
        raise ValueError("The function call should only specify either min_weight or min_percentile")

    G = g.copy()
    
    # 1rst phase
    if min_weight is not None:
        G.remove_edges_from([(u, v) for u, v, w in G.edges(data='weight') if w < min_weight])
        
    elif min_percentile is not None:
        distribution_of_weights = [w for u, v, w in G.edges(data='weight')]
        min_weight = float(np.percentile(distribution_of_weights, min_percentile))
    
        G.remove_edges_from([(u, v) for u, v, w in G.edges(data='weight') if w < min_weight])
    
    # 2nd phase
    G.remove_nodes_from(list(nx.isolates(G)))
    
    if out_filename is not None:
        nx.write_graphml_xml(G, out_filename)
    
    return G
    # ----------------- END OF FUNCTION --------------------- #


def compute_mean_audio_features(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean audio features for tracks of the same artist.

    :param tracks_df: tracks dataframe (with audio features per each track).
    :return: artist dataframe (with mean audio features per each artist).
    """

    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    compiled_tracks_df = tracks_df.groupby(['artist_id','artist_name']).agg({
        'danceability': 'mean',
        'energy': 'mean',
        'loudness': 'mean',
        'speechiness': 'mean',
        'acousticness':'mean',
        'instrumentalness':'mean',
        'liveness':'mean',
        'valence':'mean',
        'tempo':'mean',
    }).reset_index()

    compiled_tracks_df.to_csv("AudioFeatures.csv", index=False)

    return compiled_tracks_df
    # ----------------- END OF FUNCTION --------------------- #


def create_similarity_graph(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> \
        nx.Graph:
    """
    Create a similarity graph from a dataframe with mean audio features per artist.

    :param artist_audio_features_df: dataframe with mean audio features per artist.
    :param similarity: the name of the similarity metric to use (e.g. "cosine" or "euclidean").
    :param out_filename: name of the file that will be saved.
    :return: a networkx graph with the similarity between artists as edge weights.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #

    artist_ids = artist_audio_features_df['artist_id'].tolist()
    audio_features = artist_audio_features_df.drop(['artist_id', 'artist_name'], axis=1)

    if similarity == "cosine":
        similarity_matrix = cosine_similarity(audio_features)
    elif similarity == "euclidean":
        similarity_matrix = 1 / (1 + euclidean_distances(audio_features))
    else:
        print("Similarity not valid")
        return None
    
    G = nx.Graph()
    for i in range(len(artist_ids)):
        for j in range(i+1, len(artist_ids)):
            if j != i:
                weight = similarity_matrix[i, j]
                G.add_edge(artist_ids[i], artist_ids[j], weight=weight)

    if out_filename is not None:
        nx.write_graphml_xml(G, out_filename)

    return G
    # ----------------- END OF FUNCTION --------------------- #


if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #

    songs = pd.read_csv("TrackData.csv")

    G_bfs = nx.read_graphml("gB.graphml")
    print(f"Original BFS graph: {G_bfs}")
    G_dfs = nx.read_graphml("gD.graphml")
    print(f"Original DFS graph: {G_dfs}")

    # 6 A)
    G_bfs_2 = retrieve_bidirectional_edges(G_bfs, "gBp.graphml")
    G_dfs_2 = retrieve_bidirectional_edges(G_dfs, "gDp.graphml")
    
    # 6 B)
    compiled_songs = compute_mean_audio_features(songs)
    grafic = create_similarity_graph(compiled_songs, "cosine", "Similarity.graphml")
    #print(f"Similarity graph: {grafic}")

    min_percentile=98.3
    gwB = prune_low_weight_edges(grafic, out_filename="gBw.graphml", min_percentile=min_percentile)
    min_weight=0.99999805
    gwD = prune_low_weight_edges(grafic, out_filename="gDw.graphml", min_weight=min_weight)

    # 1
    print("\n - 1")
    print(f"gB': {G_bfs_2}")
    print(f"gD': {G_dfs_2}\n")
    print(f"Min. percentile: {min_percentile}")
    print(f"gBw: {gwB}")
    print(f"Min. weight: {min_weight}")
    print(f"gDw: {gwD}")
    
    # ------------------- END OF MAIN ------------------------ #
