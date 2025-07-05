def bfs(llista_vertex: list[str], llista_arestes: list[tuple(str, str)]):
    faltants = llista_vertex
    llista_explorats = []
    for vertex in llista_vertex:
        llista_explorats.append(vertex)
        for aresta in llista_arestes:
            if aresta[0] == vertex:
                llista_explorats.append(aresta[1])
                faltants.remove(aresta[1])
    llista_explorats = list(set(llista_explorats))
    return len(faltants)

print(bfs())