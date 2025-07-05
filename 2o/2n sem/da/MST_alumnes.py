import networkx as nx
import matplotlib.pyplot as plt


def creaGraf(graf):
    G = nx.Graph()
    G.add_nodes_from(list(graf.keys()))
    for k,v in graf.items():
        for arestes in v:
            G.add_edge(k, arestes[0], weight=arestes[1] )
    return G    

def pintaGraf(G):
    
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, edgecolors='red', node_color='lightgray', node_size=500)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    nx.draw_networkx_labels(G, pos)

    # Show the graph
    fig = plt.figure()
    fig = plt.axis('off')
  
    
def pintaMST(G,MST_Gr):
    
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, edgecolors='red', node_color='lightgray', node_size=500)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_edges(MST_Gr, pos,width = 2, edge_color='green')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    nx.draw_networkx_labels(G, pos)
    
    # Show the graph
    plt.figure()
    plt.axis('off')



graf_1 = {'a': [('b',6), ('c',1), ('d',5)],
         	'b': [('a',6), ('e',3), ('c',5)],
          'c': [('a',1), ('b',5), ('e',6), ('d',5),('f',4)],
			   'd': [('a',5), ('c',5),('f',2)],
			'e': [('b',3), ('f',6), ('c',6)],			
			'f': [('e',6), ('c',4),('d',2)],
		}

graf_2 = {'a': [('p',4), ('j',15), ('b',1)],
         	'b': [('a',1), ('d',2), ('e',2), ('c',2)],
			'j': [('a',15),('c',6)],
			'p': [('a',4),('d',8)],
			'd': [('b',2), ('g',3),('p',8)],
			'e': [('b',2), ('g',9), ('f',5), ('c',2),('h',4)],
			'c': [('b',2), ('e',2), ('f',5), ('i',20),('j',6)],
			'g': [('d',3), ('e',9), ('h',1)],
			'f': [('h',10), ('e',5), ('c',5),('i',2)],
			'i': [('c',20),('f',2)],
			'h': [('g',1),('e',4),('f',10)] 
		}

def add_edge(graph_dict: dict, n1, n2, w):
    if n1 != n2:
        try:
            graph_dict[n1].append((n2, w))
            graph_dict[n2].append((n1, w))
        except: 
            graph_dict[n1] = [(n2, w)]
            graph_dict[n2] = [(n1, w)]

        return graph_dict
    else:
        return graph_dict

def prim(graf: dict, start: str):

    cost = 0

    g = graf.copy()

    left_nodes = set(graf.keys())
    left_nodes.remove(start)
    visited_nodes = set(start)
          
    MST = {}

    while len(left_nodes) > 0:

        min_edges = []
        
        for node in visited_nodes:
            possible = [(n, w) for n, w in g[node] if n not in visited_nodes]
            if len(possible) > 0:
                min_edges.append((node, min(possible, key=lambda x: x[1]))) # (n1, (m2, w))
        
        min_e = min(min_edges, key=lambda x: x[1][1])

        n1 = min_e[0]
        n2 = min_e[1][0]
        w = min_e[1][1]

        MST = add_edge(MST, n1, n2, w)

        cost += min_e[1][1]

        visited_nodes.add(min_e[1][0])
        left_nodes.remove(min_e[1][0])

    return MST, cost

   
def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    x_root = find(parent, x)
    y_root = find(parent, y)

    if rank[x_root] < rank[y_root]:
        parent[x_root] = y_root
    elif rank[x_root] > rank[y_root]:
        parent[y_root] = x_root
    else:
        parent[y_root] = x_root
        rank[x_root] += 1

def kruskal(graf):
    edges = []
    for node in graf:
        for neighbor, weight in graf[node]:
            edges.append((weight, node, neighbor))

    edges.sort()

    parent = {}
    rank = {}

    for node in graf:
        parent[node] = node
        rank[node] = 0

    MST = {}
    cost = 0

    for edge in edges:
        weight, node1, node2 = edge
        x = find(parent, node1)
        y = find(parent, node2)
        #MST = add_edge(MST, node1, node2, weight)
        if x != y:
            if x in MST:
                MST[x].append((node2, weight))
            else:
                MST[x] = [(node2, weight)]

            if y in MST:
                MST[y].append((node1, weight))
            else:
                MST[y] = [(node1, weight)]
            union(parent, rank, x, y)
            cost += weight

    return MST, cost


g = graf_1
Gr = creaGraf(g)
pintaGraf(Gr)
MST_p,cost = prim(g,'a')
print("arbre Prim",MST_p)
print(cost)
MST_p_Gr = creaGraf(MST_p)
pintaMST(Gr,MST_p_Gr)


MST_k,cost = kruskal(g)
print("arbre Kruskal",MST_k)
print(cost)
MST_k_Gr = creaGraf(MST_k)
pintaMST(Gr,MST_k_Gr)
