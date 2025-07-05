import random
import networkx as nx
from itertools import combinations
from networkx import bipartite
from time import time

def build_graph():
    llista_edges = []
    G = nx.Graph()
    fitxer = open("email-Eu-core.txt","r")
    a = 5000
    for linia in fitxer:
        if a < 0:
            break
        linia = linia[:-1]
        string = linia.split(" ")
        tupla = (string[0],string[1])
        llista_edges.append(tupla)
        a -= 1
    fitxer.close()
    G.add_edges_from(llista_edges)
    return G

def simulate_mail(m,s):

    G = build_graph()
    for aresta in G.edges:
        valor1 = aresta[0]
        valor2 = aresta[1]
        
        num = random.gauss(m,s)
        
        if num >= 0:
            
            G[valor1][valor2]["weight"] = round(num)
     
    return G

  
def how_many_cliques(n,m,s):
    diccionari = {}
    
    G = simulate_mail(m, s)
   
    
    diccionari[2] = len(G.edges)
    
    for c in nx.enumerate_all_cliques(G):
        
        
        status = True
        if len(c) > 2:
            
            comb = combinations(c, 2)
            llista_arestes = []
            
            for aresta in comb:
                llista_arestes.append(aresta)
                
            for valor1, valor2, w in G.edges(data=True):
                if (valor1, valor2) in llista_arestes:
                    
                    if w["weight"] < n:
                        status = False
            
            if status == True:
                
                
                if len(c) not in diccionari:
                    diccionari[len(c)] = 1
                else:
                    diccionari[len(c)] += 1
                    
                
                
    for x in diccionari.keys():
        print("Grups de",x,"persones que s'han intercanviat almenys",n,"missatges:",diccionari[x])

def loto():
    n_opcions = int(input("Introdueix número d'opcions: "))
    total_opcions = int(input("Introdueix total d'opcions: "))
    
    combinacio = input("Introdueix la combinacio: ")
    combinacio = combinacio.split(" ")
    
    while len(combinacio) != n_opcions:
        combinacio = input("Introdueix la combinacio: ")
        combinacio = combinacio.split(" ")
    
    guanyador = False
    sortejos = 0
    
    while not guanyador:
        resultat = []
        sortejos += 1
   
        while len(resultat) < n_opcions:
            num = random.randint (0,total_opcions)
            resultat.append(str(num))
            
        
        if resultat == combinacio:
        
            guanyador = True
    print("Perquè la combinació",combinacio,"sortís guanyadora s'han necessitat",sortejos,"sortejos")
    return sortejos
        
    

def big_flat(G):
    

    n = len(G.nodes) # Nombre de vèrtexs
    
    if nx.is_planar(G): # Aquesa funció determina si el graf és pla o no
        return G # Si és pla retorna el graf sense canvis
    
    else: # Sinó

        while not nx.is_planar(G): # Mentre G no sigui un graf pla
        
            if n > 5: #Per k(3,3):
                for subgraf in combinations(G.nodes(),6): # Aquesta línia itera sobre tots els subgrafs de 6 vertexs possibles en el graf G 
                    
                    SubG = G.subgraph(subgraf) # Transforma el subgraf en un subgraf del tipus networkx 
                        
                    if bipartite.is_bipartite(SubG):  # Aquesta funció determina si el subgraf és bipartit
                            esquerra, dreta = bipartite.sets(SubG) # Separa els dos conjunts del graf bipartit
                            
                            if len(esquerra)==3 and len(dreta)==3: # Determina si les longituds dels dos conjunts és 3. Si es compleix, el subgraf és un K(3,3)
                                nodes = list(SubG.nodes) # Transforma els nodes del tipus networkx en una llista bàsica de python
                           
                                G.remove_node(nodes[0]) # Elimina un node del graf bipartit K(3,3) perquè el graf sigui pla
                              
                                
                
                
            if n > 4: #Per K(5) 
             
                for subgraf in combinations(G.nodes(),5): # Aquesta línia itera sobre tots els subgrafs de 5 vertexs possibles en el graf G 
                    
                    subG= G.subgraph(subgraf) # Transforma el subgraf en un subgraf del tipus networkx 
                    
                    if len(subG.edges())==10: # Mira si el subgraf de 5 vertexs té 10 arestes que significa que es tracta del graf complet K(5)
                        nodes = list(subG.nodes) # Transforma els nodes del tipus networkx en una llista bàsica de python
                   
                        G.remove_node(nodes[0])  # Elimina un node del graf complet K(5) perquè el graf sigui pla
                    
            
            
     
        return G # Retorna el graf modificat perquè siguiu pla


start = time()

n = 10
m = 50
s = 5

print(f"n: {n}")
print(f"m: {m}")
print(f"s: {s}")
how_many_cliques(n,m,s)

end = time()

print(f"Temps: {end - start} segons")