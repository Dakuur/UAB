import graph
import math
import sys
import queue
import dijkstra
import graph_lib as gl
from typing import List, Tuple, Dict
from copy import deepcopy, copy
import heapq

# SalesmanTrackBranchAndBound1 ===================================================


def SalesmanTrackBranchAndBound1(g: gl.Graph, visits: gl.Visits):  # NO IMPLEMENTAR
    raise Exception("Funció de l'apartat 1 no implementada")


# SalesmanTrackBranchAndBound2 ===================================================


def distances_and_paths(g: gl.Graph, visits: gl.Visits) -> Tuple[dict, dict]:
    """
    Returns the distances and paths between all vertices in the graph
    :param g: Graph
    :param visits: Visits
    :return: distances, paths
    """
    distances = {}
    paths = {}
    for vertex in visits.Vertices:  # skip the last vertex (no exiting paths from it)
        dists_v, paths_v = dijkstra.Dijkstra(g, vertex)
        distances[vertex.Name] = dists_v
        paths[vertex.Name] = paths_v
    return distances, paths


def min_max_levels(
    dist_dict: dict, path: list, visits: gl.Visits
) -> Dict[str, Tuple[float, float]]:
    """
    Returns the minimum level of the vertices in the path
    :param dist_dict: dict with the distances between all vertices
    :param path: list of vertices
    :param visits: Visits
    :return: minimum level
    """
    to_visit = [
        v.Name for v in visits.Vertices if (v not in path and v != visits.Vertices[-1])
    ]
    heuristics = dict()

    for i in to_visit:
        maximum_d = 0
        minimum_d = math.inf
        for k, v in dist_dict.items():
            if i == k:
                continue
            maximum_d = max(maximum_d, v[i])
            minimum_d = min(minimum_d, v[i])
        heuristics[i] = (minimum_d, maximum_d)

    return heuristics


def build_track(g: gl.Graph, path: list, best_paths: dict) -> gl.Track:
    """
    Builds the track (all vertices and edges) from the best paths between the vertices to be visited (ordered)
    :param g: Graph
    :param path: list of Vertices to be visited
    :param best_paths: dict with the best paths between all vertices to be visited
    """
    track = gl.Track(g)
    [
        track.AddLast(e)
        for i in range(len(path) - 1)
        for e in best_paths[path[i].Name][path[i + 1].Name]
    ]
    return track


def SalesmanTrackBranchAndBound2(g: gl.Graph, visits: gl.Visits):
    dist_dict, paths_dict = distances_and_paths(g, visits)
    best_path = []
    best_distance = math.inf

    def branch_and_bound(path: List[gl.Vertex], total_distance: float):
        nonlocal best_path, best_distance

        if len(path) == len(visits.Vertices):
            if (
                path[-1] == visits.Vertices[-1] and total_distance < best_distance
            ):  # ha arribat final
                best_distance = total_distance
                best_path = path
            return

        heuristics = min_max_levels(dist_dict, path, visits)

        # Utilizar heapq para manejar la pila de prioridades
        heap = []

        for vertex in visits.Vertices:
            if vertex not in path or (
                len(path) == len(visits.Vertices) - 1 and vertex == path[0]
            ):
                new_distance = total_distance + dist_dict[path[-1].Name][vertex.Name]

                min_heuristic = (
                    heuristics[vertex.Name][0] if vertex.Name in heuristics else 0
                )
                if new_distance + min_heuristic >= best_distance:
                    continue

                heapq.heappush(heap, (new_distance + min_heuristic, vertex))

        while heap:
            _, vertex = heapq.heappop(heap)
            branch_and_bound(
                path + [vertex], total_distance + dist_dict[path[-1].Name][vertex.Name]
            )

    start = visits.Vertices[0]
    branch_and_bound([start], 0)

    track = build_track(g, best_path, paths_dict)
    return track


# SalesmanTrackBranchAndBound3 ===================================================


def SalesmanTrackBranchAndBound3(g: gl.Graph, visits: gl.Visits):  # NO IMPLEMENTAR
    raise Exception("Funció de l'apartat 3 no implementada")
