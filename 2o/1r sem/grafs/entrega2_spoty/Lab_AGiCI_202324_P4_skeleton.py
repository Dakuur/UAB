import pandas as pd
import matplotlib.pyplot as plt
import Lab_AGiCI_202324_P1_skeleton as a
import Lab_AGiCI_202324_P2_skeleton as c
import Lab_AGiCI_202324_P3_skeleton as b
import networkx as nx
import numpy as np

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #

def get_id_artista(audio_features, nom):
    return audio_features.loc[audio_features['artist_name'] == nom, 'artist_id'].values[0]

def similar_artist(similarity_graph: nx.Graph, id_artista, ref_graph: nx.Graph):
    """
    Returns a list of artist ids sorted by their similarity to the artist given.

    :param similarity_graph: similarity graph generated in Lab_AGiCI_202324_P2_skeleton.
    :param nom_artista: string with the name of the artist to analyze.
    """
    dataframe = pd.read_csv("AudioFeatures.csv")
    
    neighbors = list(similarity_graph.neighbors(id_artista))
    neighbors_sorted = sorted(neighbors, key=lambda x: similarity_graph[id_artista][x]['weight'], reverse=True)
    neighbors_in_ref_graph = [neighbor for neighbor in neighbors_sorted if neighbor in ref_graph.nodes]

    return neighbors_in_ref_graph
    

# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def plot_degree_distribution(degree_dict: dict, normalized: bool = False, loglog: bool = False) -> None:
    """
    Plot degree distribution from dictionary of degree counts.

    :param degree_dict: dictionary of degree counts (keys are degrees, values are occurrences).
    :param normalized: boolean indicating whether to plot absolute counts or probabilities.
    :param loglog: boolean indicating whether to plot in log-log scale.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    degrees, counts = zip(*degree_dict.items())
    
    if normalized == True:
        norm = sum(counts)
        counts = [count/norm for count in counts]
    
    plt.figure(figsize=(8, 6))

    if loglog == True:
        plt.loglog(degrees, counts, 'o', markersize=10)
        plt.xlabel('Degree')
        plt.ylabel('Relative Frequency' if normalized else 'Frequency')
        
    else:
        plt.plot(degrees, counts, 'o', markersize=10)
        plt.xlabel('Degree')
        plt.ylabel('Relative Frequency' if normalized else 'Frequency')

    plt.title('Degree Distribution')
    plt.show()
        
    # ----------------- END OF FUNCTION --------------------- #


def plot_audio_features(artists_audio_feat: pd.DataFrame, artist1_id: str, artist2_id: str) -> None:
    """
    Plot a (single) figure with a plot of mean audio features of two different artists.

    :param artists_audio_feat: dataframe with mean audio features of artists.
    :param artist1_id: string with id of artist 1.
    :param artist2_id: string with id of artist 2.
    :return: None
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    f1 = artists_audio_feat.loc[artists_audio_feat['artist_id'] == artist1_id].iloc[:, 2:]
    n1 = artists_audio_feat.loc[artists_audio_feat['artist_id'] == artist1_id].iloc[:,1].values[0]
    
    f2 = artists_audio_feat.loc[artists_audio_feat['artist_id'] == artist2_id].iloc[:, 2:]
    n2 = artists_audio_feat.loc[artists_audio_feat['artist_id'] == artist2_id].iloc[:,1].values[0]
    
    plt.figure(figsize=(10, 6))
    width = 0.35
    x = ["danceability","energy","loudness","speechiness","acousticness","instrumentalness","liveness","valence","tempo"]
    
    v1a = f1.values.tolist()[0]
    v2a = f2.values.tolist()[0]
    suma = [x+y for x,y in zip(v1a,v2a)]
    
    v1_norm = [x/num for x,num in zip(v1a, suma)]
    v2_norm = [x/num for x,num in zip(v2a, suma)]
    
    X = np.arange(len(x)) 

    plt.bar(X-0.2, v1_norm, 0.4, label = n1) 
    plt.bar(X+0.2, v2_norm, 0.4, label = n2) 
    
    plt.xticks(X, x, fontsize = 5)
    plt.xlabel('Audio Features')
    plt.ylabel('Values')
    plt.title('Comparison of Audio Features from Two Artists')
    
    plt.legend()
    plt.show()
    
    # ----------------- END OF FUNCTION --------------------- #


def plot_similarity_heatmap(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> None:
    """
    Plot a heatmap of the similarity between artists.

    :param artist_audio_features_df: dataframe with mean audio features of artists.
    :param similarity: string with similarity measure to use.
    :param out_filename: name of the file to save the plot. If None, the plot is not saved.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    graph = c.create_similarity_graph(artist_audio_features_df, similarity)
    
    
    nodes = list(graph.nodes())
    edge_weights = np.array([[graph[u][v]['weight'] for v in nodes if u!=v] for u in nodes])

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(edge_weights, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Similarity')
    plt.xticks(range(len(nodes)), nodes, rotation=90, fontsize = 2)
    plt.yticks(range(len(nodes)), nodes, fontsize = 2)
    plt.title('Similarity Heatmap from Graph Weights')

    # Show the plot or save to a file
    if out_filename:
        plt.savefig(out_filename, bbox_inches='tight')
    else:
        plt.show()
    # ----------------- END OF FUNCTION --------------------- #


if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    
    df = pd.read_csv("AudioFeatures.csv")

    id_taylor = get_id_artista(df, "Taylor Swift")
    
    similarity = nx.read_graphml("Similarity.graphml")
    reference_graph = nx.read_graphml("gB.graphml")

    similar = similar_artist(similarity, id_taylor, reference_graph)

    # 4 a)
    files_to_plot = ["gBp.graphml", "gDp.graphml", "gBw.graphml"]
    for i in files_to_plot:
        g = nx.read_graphml(i) # graph
        dicc = b.get_degree_distribution(g)
        plot_degree_distribution(dicc, normalized = True)

    # 4 b)
    most_similar = similar[0]
    plot_audio_features(df, id_taylor, most_similar)

    # 4 c)
    least_similar = similar[-1]
    plot_audio_features(df, id_taylor, least_similar)

    # 4 d)
    plot_similarity_heatmap(df, "cosine")
    plot_similarity_heatmap(df, "euclidean")

    # ------------------- END OF MAIN ------------------------ #
