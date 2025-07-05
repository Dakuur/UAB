import graph
import math
import sys
import queue

# Dijkstra =====================================================================


def Dijkstra(g, start):
    for vertex in g.Vertices:
        vertex.DijkstraDistance = math.inf
        vertex.DijkstraVisit = False
        vertex.set_previous_edge(
            None
        )  # Inicializa la arista previa en el camino más corto como None

    start.DijkstraDistance = 0

    current_vertex = start
    paths = {v.Name: [] for v in g.Vertices}

    while True:
        current_vertex.DijkstraVisit = True

        for edge in current_vertex.Edges:
            neighbor_vertex = edge.Destination

            new_distance = current_vertex.DijkstraDistance + edge.Length

            if new_distance < neighbor_vertex.DijkstraDistance:
                neighbor_vertex.DijkstraDistance = new_distance
                neighbor_vertex.set_previous_edge(edge)
                paths[neighbor_vertex.Name] = paths[current_vertex.Name] + [edge]

        min_distance = math.inf
        next_vertex = None
        for vertex in g.Vertices:
            if not vertex.DijkstraVisit and vertex.DijkstraDistance < min_distance:
                min_distance = vertex.DijkstraDistance
                next_vertex = vertex

        if next_vertex is None:
            break

        current_vertex = next_vertex

    distances = dict()

    for vertex in g.Vertices:
        if vertex.DijkstraDistance == math.inf:
            vertex.DijkstraDistance = sys.float_info.max
        distances[vertex.Name] = vertex.DijkstraDistance

    return distances, paths


# DijkstraQueue ================================================================


def DijkstraQueue(g, start):
    for vertex in g.Vertices:
        vertex.DijkstraDistance = math.inf
        vertex.DijkstraVisit = False
        vertex.set_previous_edge(None)

    start.DijkstraDistance = 0

    vertex_queue = queue.PriorityQueue()

    vertex_queue.put((0, start))

    while not vertex_queue.empty():
        current_distance, current_vertex = vertex_queue.get()

        if current_vertex.DijkstraVisit:
            continue

        current_vertex.DijkstraVisit = True

        for edge in current_vertex.Edges:
            neighbor_vertex = edge.Destination

            new_distance = current_vertex.DijkstraDistance + edge.Length

            if new_distance < neighbor_vertex.DijkstraDistance:
                neighbor_vertex.DijkstraDistance = new_distance
                neighbor_vertex.set_previous_edge(edge)

                vertex_queue.put((new_distance, neighbor_vertex))

    for vertex in g.Vertices:
        if vertex.DijkstraDistance == math.inf:
            vertex.DijkstraDistance = 1.7976931348623157081e308
