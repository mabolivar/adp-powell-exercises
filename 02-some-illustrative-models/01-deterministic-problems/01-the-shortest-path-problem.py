import time
from collections import defaultdict

GRAPH_PATH = "data/01-spp-graph.dat"


def load_graph(path):
    with open(path) as file:
        next(file, None)
        arcs = {tuple([i, j]): c for line in file for i, j, c in [map(int, line.split(','))]}

    nodes = set()
    fstar = defaultdict(list)
    bstar = defaultdict(list)
    for i, j in arcs.keys():
        nodes.add(i)
        nodes.add(j)
        fstar[i].append(j)
        bstar[j].append(i)

    return nodes, arcs, fstar, bstar


def get_path(node, predecesors):
    path = list()
    p = get_path(predecesors[node], predecesors) if node in predecesors.keys() else []
    path.append(node)
    return path + p


def get_shortest_path(nodes, arcs, bstar, start, end):
    M = 999
    v = {i: M if i != end else 0 for i in nodes}
    p = {i: M for i in nodes if i != end}
    top_candidates = [end]

    while len(top_candidates) > 0:
        j = top_candidates.pop()
        for i in bstar[j]:
            if v[i] > arcs[(i, j)] + v[j]:
                v[i] = arcs[(i, j)] + v[j]
                p[i] = j
                top_candidates.append(i) if i not in top_candidates else None
        print("Current node:\t" + str(j) + "\tv:\t" + str(v))
    return v[start], get_path(start, p)


if __name__ == "__main__":
    now = time.strftime('%d/%m/%Y %H:%M:%S')
    nodes, arcs, _, bstar = load_graph(GRAPH_PATH)
    cost, path = get_shortest_path(nodes, arcs, bstar, min(nodes), max(nodes))
    print("--------------------------")
    print("Current time:\t" + now)
    print("--------------------------")
    print('Objective function:\t' + str(cost))
    print('Path:\t' + str(path))
    print("--------------------------")
