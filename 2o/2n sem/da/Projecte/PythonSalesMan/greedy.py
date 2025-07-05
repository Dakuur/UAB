import graph
import math
import sys
import queue
import dijkstra

# SalesmanTrackGreedy ==========================================================


def SalesmanTrackGreedy(g, visits):
    result_track = graph.Track(g)

    v = visits.Vertices[0]

    candidatos = visits.Vertices[1:]

    while candidatos:

        dijkstra.Dijkstra(g, v)
        v1 = min(candidatos, key=lambda vertex: vertex.DijkstraDistance)

        current_vertex = v1
        while current_vertex != v:
            edge = current_vertex.PreviousEdge
            result_track.AddFirst(edge)
            current_vertex = edge.Origin

        candidatos.remove(v1)

        v = v1

    dijkstra.Dijkstra(g, v)
    final_vertex = visits.Vertices[-1]
    current_vertex = final_vertex
    while current_vertex != v:
        edge = current_vertex.PreviousEdge
        result_track.AddFirst(edge)
        current_vertex = edge.Origin

    return result_track
