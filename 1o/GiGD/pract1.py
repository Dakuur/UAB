import networkx as nx
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import scipy
import pandas as pd

def build_simplegraph():
    G = nx.Graph()
    fitxer = open("GiGD/email-Eu-core.txt", "r")
    llistaarestes = []
    for linia in fitxer:
        aresta = linia.split()
        tupla = (aresta[0],aresta[1])
        llistaarestes.append(tupla)
    G.add_edges_from(llistaarestes)
    fitxer.close()
    return G

G = build_simplegraph()

#print(G.edges)
#nx.draw(G, with_labels=True)
#nx.draw_kamada_kawai(G, with_labels=True)

def how_many_degrees(G: nx.Graph, a, b):
    iteracions = 0
    current_nodes = set()
    current_nodes.add(a)
    visited_nodes = set()

    while b not in current_nodes:
        iteracions += 1
        next_nodes = set()
        for node in current_nodes:
            if node not in visited_nodes:
                next_nodes.update(G.neighbors(node))
                visited_nodes.add(node)
        current_nodes = next_nodes
    return iteracions

print(how_many_degrees(G, "1", "500"))