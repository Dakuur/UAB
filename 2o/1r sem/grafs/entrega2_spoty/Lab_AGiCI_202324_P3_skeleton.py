import networkx as nx
import pandas as pd
import Lab_AGiCI_202324_P2_skeleton as c

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #


# --------------- END OF AUXILIARY FUNCTIONS ------------------ #

def num_common_nodes(*arg):
    """
    Return the number of common nodes between a set of graphs.

    :param arg: (an undetermined number of) networkx graphs.
    :return: an integer, number of common nodes.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    if len(arg) < 2:
        raise ValueError ("S'han de donar mínim dos grafs")
        
    set_graph = arg[0]
    
    for i in range(1,len(arg)):
        set_graph = nx.intersection(set_graph, arg[i])
        
    return len(list(set_graph.nodes))
    # ----------------- END OF FUNCTION --------------------- #


def get_degree_distribution(g: nx.Graph) -> dict:
    """
    Get the degree distribution of the graph.

    :param g: networkx graph.
    :return: dictionary with degree distribution (keys are degrees, values are number of occurrences).
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    dict = {}
    for degree in g.degree():
        if degree[1] not in dict.keys():
            dict[degree[1]] = 1
            
        else:
            dict[degree[1]] += 1
            
    return dict
    # ----------------- END OF FUNCTION --------------------- #


def get_k_most_central(g: nx.Graph, metric: str, num_nodes: int) -> list:
    """
    Get the k most central nodes in the graph.

    :param g: networkx graph.
    :param metric: centrality metric. Can be (at least) 'degree', 'betweenness', 'closeness' or 'eigenvector'.
    :param num_nodes: number of nodes to return.
    :return: list with the top num_nodes nodes with the specified centrality.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    if metric == 'degree':
        score = nx.degree_centrality(g)
    elif metric == 'betweenness':
        score = nx.betweenness_centrality(g)
    elif metric == 'closeness':
        score = nx.closeness_centrality(g)
    elif metric == 'eigenvector':
        score = nx.eigenvector_centrality(g)
    else:
        raise ValueError("Centrality metric not implemented")
        
    return (sorted(score, key = score.get, reverse = True))[:num_nodes]
    # ----------------- END OF FUNCTION --------------------- #


def find_cliques(g: nx.Graph, min_size_clique: int) -> tuple:
    """
    Find cliques in the graph g with size at least min_size_clique.

    :param g: networkx graph.
    :param min_size_clique: minimum size of the cliques to find.
    :return: two-element tuple, list of cliques (each clique is a list of nodes) and
        list of nodes in any of the cliques.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #

    all_cliques = list(nx.find_cliques(g))

    cliques_sized = [clique for clique in all_cliques if len(clique) >= min_size_clique]
    
    nodes = set(node for clique in cliques_sized for node in clique)
    
    return cliques_sized, list(nodes)
    # ----------------- END OF FUNCTION --------------------- #


def detect_communities(g: nx.Graph, method: str) -> tuple:
    """
    Detect communities in the graph g using the specified method.

    :param g: a networkx graph.
    :param method: string with the name of the method to use. Can be (at least) 'givarn-newman' or 'louvain'.
    :return: two-element tuple, list of communities (each community is a list of nodes) and modularity of the partition.
    """
    # ------- IMPLEMENT HERE THE BODY OF THE FUNCTION ------- #
    
    if method.lower() == 'girvan-newman':
        communities = nx.community.girvan_newman(g)
        
    elif method.lower() == 'louvain':
        communities = nx.community.louvain_communities(g)
        
    else:
        raise ValueError ("No method was found")

    communities = list(communities)
    mod_list = [nx.community.modularity(g, community) for community in communities]

    return communities, mod_list
    
    # ----------------- END OF FUNCTION --------------------- #



if __name__ == '__main__':
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #

    """a = nx.Graph()
    a.add_edges_from([(1,2),(1,3),(3,2),(1,5),(5,4),(5,6)])
    b = nx.Graph()
    b.add_edges_from([(1,2),(1,5)])
    
    #print(num_common_nodes(a,b))
    #print(get_degree_distribution(a))
    #print(get_k_most_central(a, 'degree', 3))
    z = (detect_communities(a,'givarn-newman'))"""
    
    songs = pd.read_csv("TrackData.csv")

    G_bfs = nx.read_graphml("gB.graphml")
    print(f"Original BFS graph: {G_bfs}")
    G_dfs = nx.read_graphml("gD.graphml")
    print(f"Original DFS graph: {G_dfs}")

    # 6 A)
    G_bfs_2 = c.retrieve_bidirectional_edges(G_bfs, "gBp.graphml")
    G_dfs_2 = c.retrieve_bidirectional_edges(G_dfs, "gDp.graphml")
    
    # 6 B)
    compiled_songs = c.compute_mean_audio_features(songs)
    grafic = c.create_similarity_graph(compiled_songs, "cosine", "Similarity.graphml")
    #print(f"Similarity graph: {grafic}")

    min_percentile=98.3
    gwB = c.prune_low_weight_edges(grafic, out_filename="gBw.graphml", min_percentile=min_percentile)
    min_weight=0.99999805
    gwD = c.prune_low_weight_edges(grafic, out_filename="gDw.graphml", min_weight=min_weight)
    
    # ------------------ 1 ------------------
    print(" - 1)")
    nodes_gb_gd = num_common_nodes(G_bfs, G_dfs)
    nodes_gbnd_gdnd = num_common_nodes(G_bfs_2, G_dfs_2)
    nodes_gwb_gwd = num_common_nodes(gwB, gwD)
    print("Num. de nodes compartits entre els grafs:")
    print("\tgB i gD:", nodes_gb_gd)
    print("\tgB' i gD':", nodes_gbnd_gdnd)
    print("\tgBw i gDw:", nodes_gwb_gwd)
    
    nodes_gb_gbnd = num_common_nodes(G_bfs, G_bfs_2)
    print("\tgB i gB':", nodes_gb_gbnd)
    
    nodes_gwb_gbnd = num_common_nodes(gwB, G_bfs_2)
    print("\tgBw i gB':", nodes_gwb_gbnd)
    
    n = num_common_nodes(G_bfs,G_dfs,G_bfs_2,G_dfs_2,gwB,gwD)
    print("\tTots els grafs:", n)
    
    # ------------------ 2 ------------------
    print(" - 2)")
    degree_centralities = nx.degree_centrality(G_bfs_2)
    betweenness_centralities = nx.betweenness_centrality(G_bfs_2)

    top_25_degree = [node for node, x in sorted(degree_centralities.items(), key=lambda x: x[1], reverse=True)[:25]]
    top_25_betweenness = [node for node, x in sorted(betweenness_centralities.items(), key=lambda x: x[1], reverse=True)[:25]]

    common =  set(top_25_degree).intersection(top_25_betweenness)
    print(f"25 highest degree centrality: \n{top_25_degree}")
    print(f"25 highest betweeness centrality: \n{top_25_betweenness}")
    print(f"Nodes en comú als 2 tops: {len(common)}")
    
    # ------------------ 3 ------------------
    print(" - 3)")
    cliques_B, nodes_B = find_cliques(G_bfs_2, min_size_clique=7)

    cliques_D, nodes_D = find_cliques(G_dfs_2, min_size_clique=6)

    print("Cliques a gB':", len(cliques_B))
    print("Cliques a gD':", len(cliques_D))

    nodes_unics_gbnd = set(node for node in nodes_B)
    
    nodes_unics_gdnd = set(node for node in nodes_D)
    
    print("Nodes únics a gB': ",len(nodes_unics_gbnd))
    print("Nodes únics a gD': ",len(nodes_unics_gdnd))
    
    nodes_comuns_entre_grafs = nodes_unics_gbnd.intersection(nodes_unics_gdnd)
    print("Nodes en comú entre gB' i gD': \n",len(nodes_comuns_entre_grafs))
    
    # ------------------ 4 ------------------
    print(" - 4)")
    max_size_clique_gbnd = max(cliques_B, key=len)
    print("Mida màxima de la clique a gB':", len(max_size_clique_gbnd))

    df = pd.read_csv('TrackData.csv')
    llista_cliques, llista_nodes = find_cliques(G_bfs_2, 8)
    audio_df = pd.read_csv('AudioFeatures.csv')
    audio_df_filtrat = audio_df[audio_df['artist_id'].isin(llista_nodes)]
    
    # ------------------ 5 ------------------
    print(" - 5)")
    try:
        communities, modularity = detect_communities(G_dfs, method='girvan-newman')
        print("Num. de comunitats:", len(communities))
        print("Modularitat:", str(modularity))
    except:
        pass

    # ------------------- END OF MAIN ------------------------ #