import networkx as nx
import matplotlib as mpl
from matplotlib import pyplot as plt
from time import time
#import numpy as np
#import scipy
#import pandas as pd

#1

def build_digraph():
    
    #La funció crea un graf dirigit
    
    llista_edges = [] #Crea una llista per les arestes
    G = nx.DiGraph() #Genera un graf del tipus dirigit
    fitxer = open("email-Eu-core.txt","r") #Obre l'arxiu que conté els nodes amb les connexions
    
    for linia in fitxer:
        linia = linia[:-1] #Treure el \n
        string = linia.split(" ") #Separar la linia
        tupla = (string[0],string[1]) #Trasnformar en una tupla fer afegir-la com una aresta
        llista_edges.append(tupla) #Afegeix la aresta
        
        #Transforma cada línia en una aresta pel graf
    
    fitxer.close() #Tanca el fitxer
    G.add_edges_from(llista_edges)
    #nx.draw(G, with_labels=True) #Utilitzat per graficar el graf
    #plt.show() #Utilitzat per graficar el graf
    return G #Retorna el graf dirigit

#1.1

def build_simplegraph():
   
    #La funció crea un graf no dirigit
   
    llista_edges = [] #Crea una llista per les arestes
    G = nx.DiGraph() #Genera un graf del tipus dirigit
    fitxer = open("email-Eu-core.txt","r") #Obre l'arxiu que conté els nodes amb les connexions
   
    for linia in fitxer: #Per cada línia del fitxer
        linia = linia[:-1] #Treure el \n
        string = linia.split(" ") #Separar la linia
        tupla = (string[0],string[1]) #Trasnformar en una tupla fer afegir-la com una aresta
        llista_edges.append(tupla) #Afegeix la aresta
   
    fitxer.close() #Tanca el fitxer
    G.add_edges_from(llista_edges) #Afegeix les arestes de la llista ja creada
   
    return G #Retorna el graf simple

#2

def components_DFS(G: nx.Graph):

    llista_vertexs = G.nodes #Llista de tots els vèrtexs
    llista_arestes = G.edges #Llista de totes les arestes
    used = [] #Llista que guardarà els vertexs ja marcats
    llista_components = [] #Llista que serà el return de la funció (retorna una llista de llistes on cada element pertany a un component)
    llista_actual = [] #Llista per guardar la búsqueda d'un sol component
    
    for vertex in G.nodes: #Per cada node de G
        if vertex not in used: #Comprova que el vèrtex no estigui marcat
            queue = [vertex] #cua que guardarà els vertexs que s'utilitzaran per buscar, en el cas DFS sempre s'agafarà l'ultim vertèx per arribar fins al final del camí
                        #d'aquest, abans de retrocedir
            while len(queue) > 0: #Mentres la cua no estigui buida
                vertex = queue.pop() #S'agafa l'últim vèrtex
                if vertex not in used: #Si el vèrtex no està marcat
                    used.append(vertex) #Es marca el vèrtex
                    llista_actual.append(vertex) #S'afegeix el vèrtex a la búsqueda actual
                    for y in G.neighbors(vertex): #Per cada vèrtex veí del vèrtex input
                        if y not in used: #Si no està marcat
                            queue.append(y) #s'afegeix el vèrtex veí a la cua
                  
            llista_components.append(llista_actual) #Afegeix la búsqueda d'un component de G sencer a la llista final
            llista_actual = [] #Inicialitzem la llista per buscar una component diferent

    return llista_components #Retorna la llista de les components (una llista de llistes on cada element pertany a un component de G)

#3

def components_BFS(G: nx.Graph):
    
    llista_vertexs = G.nodes #Llista de tots els vèrtexs
    llista_arestes = G.edges #Llista de totes les arestes
    used = [] #Llista que guardarà els vertexs ja marcats
    llista_components = [] #Llista que serà el return de la funció (retorna una llista de llistes on cada element pertany a un component)
    llista_actual = [] #Llista per guardar la búsqueda d'un sol component
     
    for vertex in G.nodes: #Per cada node de G
        if vertex not in used: #Comprova que el vèrtex no estigui marcat
            queue = [vertex] #cua que guardarà els vertexs que s'utilitzaran per buscar, en el cas BFS sempre s'agafarà el primer vertèx
                              #per veure tots els vertexs veins del primer vertex, abans de fer un altre salt
            
            while len(queue) > 0: #Mentres la cua no estigui buida
                vertex = queue.pop(0) #S'agafa el primer vèrtex de la cua
                if vertex not in used: #Si el vèrtex no està marcat
                     used.append(vertex) #Es marca el vèrtex
                     llista_actual.append(vertex) #S'afegeix el vèrtex a la búsqueda actual
                for y in G.neighbors(vertex): #Per cada vèrtex veí del vèrtex input
                    if y not in used: #Si no està marcat
                        queue.append(y) #s'afegeix el vèrtex veí a la cua
                        used.append(y) #Es marca el vèrtex
                        llista_actual.append(y) #S'afegeix el vèrtex a la búsqueda actual

            llista_components.append(llista_actual) #Afegeix la búsqueda d'un component de G sencer a la llista final
            llista_actual = [] #Inicialitzem la llista per buscar una component diferent
             
    return llista_components #Retorna la llista de les components (una llista de llistes on cada element pertany a un component de G)

#4

def how_many_degrees(G: nx.Graph, a, b):
    iteracions = 0 #Variable que guarda el nombre de "passos" que fa falta per arribar a b des de a
    current_nodes = set() 
    current_nodes.add(a) #Guardem els nodes que estem analitzant en un coujunt, per millor temps de còmput
    visited_nodes = set() #fem el mateix amb els node que ja hem visitat i no ens interessa calcular

    while b not in current_nodes: #cada cop mira si, entre els nodes que està analitzant, hi ha el que volem trobar
        iteracions += 1 #incrementem si no l'ha trobat
        next_nodes = set() #fa un "reset" dels pròxims nodes que analitzarà
        for node in current_nodes: #itera en els nodes que s'estan analitzant
            if node not in visited_nodes: #si el node actual no està a la llista dels que volem descartar
                next_nodes.update(G.neighbors(node)) #suma a la llista de nodes per analitzar
                visited_nodes.add(node) #afegeix al conjunt de visitats el node actual
        current_nodes = next_nodes #assigna els nodes que es faran servir al pròxim pas als next_nodes
    return iteracions #retorna el nombre de passes que ha necessitat 

def diametres(G: nx.Graph):
    llistadiametres = [] #llista que guardarà tots els diàmetres de cada parella de nodes
    llista_components = components_DFS(G)
    #el que farem serà calcular cada distància basant-nos en una matriu triangular com aquesta:
    '''
    (0 X X X)
    (/ 0 X X)
    (/ / 0 X)
    (/ / / 0)
    '''
    #fem servir files com a node 1 i columnes com node 2 (que son els que compararem amb el how_many_degrees())
    #sabem que la diagonal és 0, ja que sabem que la distància entre un node i ell mateix és 0
    #com que, en un graf simple, sabem que la distància entre dos nodes x, y és la mateixa que entre y, x, només calcularem una d'elles

    for i in llista_components:
        maxcomponent = 0
        llista_nodes = i
        for node in i: #iterem per cada node (files)
            maxnode = 0
            for node2 in llista_nodes: #iterem per cada node (columnes)
                #print(f"Calculant distància entre {node} i {node2}") #control
                distancia = how_many_degrees(G, node, node2)
                if distancia > maxnode:
                    maxnode = distancia
            if maxnode > maxcomponent:
                maxcomponent = maxnode
            llista_nodes = llista_nodes[1:] #treiem el primer node de la llista per a cumplir amb l'estructura de la matriu triangular
        llistadiametres.append(maxcomponent) #afegim la màxima d'entre les mínimes distàncies generades, que és el diàmetre del node
    return llistadiametres #retorna la llista de diàmetres per cada node

#Main:

G = build_simplegraph()

nx.draw(G, with_labels=True, node_size = 200)
#nx.draw_kamada_kawai(G, with_labels=True)

print(f"Vèrtexs: {len(G.nodes)}")
print(f"Arestes: {len(G.edges)}")

temps_inici = time()
components_BFS(G)
temps_final = time()
print("Temps BFS: ",temps_final-temps_inici)

temps_inici = time()
components_DFS(G)
temps_final = time()
print("Temps DFS: ",temps_final-temps_inici)

print(how_many_degrees(G, "300", "800"))

diametres_llista = diametres(G)
print(f"Diàmetres: {diametres_llista}")