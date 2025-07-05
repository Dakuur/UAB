import graph_lib as gl
import graph
import math
import sys
import queue
import dijkstra
import copy

# ============================= BACKTRACKING PUR =============================


def explore_next(
    current_v: gl.Vertex,
    current_e: gl.Edge,
    visits: list,
    g: gl.Graph,
    final: gl.Vertex,
):
    global current_dist, current_path, optimal_dist, optimal_path

    prev_list = current_e.Destination.Prev

    if (
        current_v not in prev_list
        and current_dist + current_e.Length < optimal_dist
        and len(prev_list) <= 1
        and (current_e.Destination in visits or len(current_e.Destination.Edges) > 1)
    ):

        prev_list.append(current_v)

        current_path.append(current_e)
        current_dist += current_e.Length

        done = False
        if current_e.Destination in visits:
            visits.remove(current_e.Destination)
            done = True

        # CRIDA A LA RECURSIVA
        backtrack_recursive(g, visits, current_e.Destination, final)

        if done:
            visits.append(current_e.Destination)

        # PAS ENRERE (LÍNIES 19-22)
        prev_list.pop()

        current_path.pop()
        current_dist -= current_e.Length


def backtrack_recursive(
    g: gl.Graph, visits: list, current_v: gl.Vertex, final: gl.Vertex
):
    global current_dist, current_path, optimal_dist, optimal_path

    if current_v == final and len(visits) == 0 and current_dist < optimal_dist:

        optimal_dist = current_dist
        optimal_path = copy.copy(current_path)

    else:
        [
            explore_next(current_v, current_e, visits, g, final)
            for current_e in sorted(current_v.Edges, key=lambda x: x.Length)
        ]


def SalesmanTrackBacktracking(g: gl.Graph, visits: list):
    global current_dist, current_path, optimal_dist, optimal_path

    # INIT
    optimal_dist = math.inf
    optimal_path = []

    current_dist = 0
    current_path = []

    for vertex in g.Vertices:
        vertex.Prev = []

    v = visits.Vertices

    backtrack_recursive(g, v[1:], v[0], v[-1])

    result_graph = graph.Track(g)
    result_graph.Edges = optimal_path

    return result_graph


# ============================ BACKTRACKING GREEDY ============================


def floyd(
    g: gl.Graph, visits: gl.Visits
):  #  fem servir diccionaris en comptes de matriu
    distances = {}
    paths = {}
    for vertex in visits.Vertices:
        dists_v, paths_v = dijkstra.Dijkstra(g, vertex)
        distances[vertex.Name] = dists_v
        paths[vertex.Name] = paths_v
    return distances, paths


def build_track(g: gl.Graph, path: list, best_paths: dict):
    track = graph.Track(g)
    [
        track.AddLast(e)
        for i in range(len(path) - 1)
        for e in best_paths[path[i].Name][path[i + 1].Name]
    ]
    return track


def SalesmanTrackBacktrackingGreedy(g: gl.Graph, visits: gl.Visits):
    best_distances, best_paths_dict = floyd(g, visits)
    best_path = []
    best_distance = math.inf

    def visit(path, total_distance):
        nonlocal best_path, best_distance
        if len(path) >= len(visits.Vertices):
            if path[-1] == visits.Vertices[-1] and total_distance < best_distance:
                best_distance = total_distance
                best_path = path
        else:
            for vertex in visits.Vertices:
                if vertex not in path or (len(path) == len(visits.Vertices) - 1 and vertex == path[0]):  # permet cicles
                    new_distance = (
                        total_distance + best_distances[path[-1].Name][vertex.Name]
                    )
                    visit(path + [vertex], new_distance)

    start = visits.Vertices[0]
    end = visits.Vertices[-1]
    visit([start], 0)

    track = build_track(g, best_path, best_paths_dict)
    return track
