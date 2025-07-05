from numpy import inf

# This class represent a graph
class Graph:

    # Initialize the class
    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = {}
        self.directed = directed

    # Add a link from A and B of given distance, and also add the inverse link if the graph is undirected
    def connect(self, A, B, distance=1):
        self.graph_dict.setdefault(A, {})[B] = distance

    # Get neighbors or a neighbor
    def get(self, a, b=None):
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    # Return a list of nodes in the graph
    def nodes(self):
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)

# This class represent a node
class Node:

    # Initialize the class
    def __init__(self, name: str, parent: str):
        self.name = name
        self.parent = parent
        self.g = 0 # Distance to start node
        self.h = 0 # Distance to goal node
        self.f = 0 # Total cost

    # Compare nodes
    def __eq__(self, other):
        return self.name == other.name

    # Sort nodes
    def __lt__(self, other):
         return self.f < other.f

    # Print node
    def __repr__(self):
        return ('({0},{1})'.format(self.name, self.f))
    
    def __hash__(self) -> int: # afegit per a treballar amb Nodes i conjunts
        return hash(self.name)

# Best-first search
def best_first_search(graph: Graph, heuristics: dict, start: str, end: str):
    next_nodes = []
    visitats = set()

    start_node = Node(start, None)
    end_node = Node(end, None)

    next_nodes.append(start_node)

    while len(next_nodes) > 0: # no necessariament passa per tots els nodes (F es innecessari aqui)
        current = min(next_nodes)
        next_nodes.remove(current)
        visitats.add(current)

        if current == end_node:
            path = []
            while current: # mentre no sigui None (fins que arribi fins arrel = start)
                path.append(current.name)
                current = current.parent #camí cap enrere (després el retorna en revers)
            return path[::-1] # trobat (retorna el primer que troba). fi del algoritme

        veins = graph.get(current.name)
        for vei_nom, dist in veins.items():
            vei = Node(vei_nom, current)
            vei.g = current.g + dist
            vei.h = heuristics.get(vei_nom, 0)
            vei.f = vei.g + vei.h

            if vei in visitats:
                continue

            if vei not in next_nodes:
                next_nodes.append(vei)
            else:
                existing_neighbor = next(n for n in next_nodes if n == vei)
                if vei.g < existing_neighbor.g:
                    existing_neighbor.g = vei.g
                    existing_neighbor.parent = current

    return None
            

# The main entry point for this module
def main():

    # Create a graph
    graph = Graph()

    # Create graph connections (Actual distance)
    graph.connect('A', 'B', 4)
    graph.connect('A', 'C', 3)
    graph.connect('B', 'F', 5)
    graph.connect('B', 'E', 12)
    graph.connect('C', 'D', 7)
    graph.connect('C', 'E', 10)
    graph.connect('D', 'E', 2)
    graph.connect('F', 'Z', 16)
    graph.connect('E', 'Z', 5)
    

    # Create heuristics (straight-line distance, air-travel distance)
    heuristics = {}
    heuristics['A'] = 14
    heuristics['B'] = 12
    heuristics['C'] = 11
    heuristics['D'] = 6
    heuristics['E'] = 4
    heuristics['F'] = 11
    heuristics['Z'] = 0


    # Run search algorithm
    path = best_first_search(graph, heuristics, 'A', 'Z')
    print(path)
    print()

# Tell python to run main method
if __name__ == "__main__": main()