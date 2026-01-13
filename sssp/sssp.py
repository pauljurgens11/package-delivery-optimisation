import heapq
import math
import time
from collections import defaultdict
from typing import List, Tuple, Dict

GraphAdj = List[List[Tuple[int, int]]]


def load_graph(path: str) -> Tuple[int, int, int, GraphAdj]:
    with open(path, "r", encoding="utf-8") as f:
        n, m, source = map(int, f.readline().split())
        adj: GraphAdj = [[] for _ in range(n)]
        for line in f:
            if not line.strip():
                continue
            u, v, w = map(int, line.split())
            adj[u].append((v, w))
    return n, m, source, adj


# -------------------------------------------------
# Range-based Single-Source Shortest Paths (SSSP)
# -------------------------------------------------

def range_sssp(
    n: int,
    source: int,
    adj: GraphAdj,
    delta: int = None,
) -> Tuple[List[float], Dict[str, float]]:
    """
    Range-based SSSP (bounded-distance Dijkstra).

    delta:
        Size of distance bands. If None, chosen automatically.
    """

    INF = float("inf")

    if delta is None:
        # Heuristic: scales well for large graphs
        delta = max(64, int(math.log2(n) ** 2))

    d = [INF] * n
    finished = [False] * n
    d[source] = 0.0

    # Buckets keyed by distance band index
    buckets: Dict[int, List[int]] = defaultdict(list)
    buckets[0].append(source)

    edge_relaxations = 0
    max_frontier_size = 0

    start_time = time.perf_counter()

    current_band = 0

    while buckets:
        # Find the smallest non-empty band
        if current_band not in buckets:
            current_band = min(buckets.keys())

        band_vertices = buckets.pop(current_band)

        # Local Dijkstra within this band
        pq = []
        for u in band_vertices:
            if not finished[u]:
                heapq.heappush(pq, (d[u], u))

        max_frontier_size = max(max_frontier_size, len(pq))

        band_limit = (current_band + 1) * delta

        while pq:
            du, u = heapq.heappop(pq)
            if finished[u]:
                continue
            if du >= band_limit:
                # Belongs to a future band
                next_band = int(du // delta)
                buckets[next_band].append(u)
                continue

            # Finalize u
            finished[u] = True

            for v, w in adj[u]:
                edge_relaxations += 1
                nd = du + w
                if nd < d[v]:
                    d[v] = nd
                    band = int(nd // delta)
                    buckets[band].append(v)
                    if band == current_band:
                        heapq.heappush(pq, (nd, v))

        current_band += 1

    runtime_sec = time.perf_counter() - start_time

    return d, {
        "runtime_sec": runtime_sec,
        "edge_relaxations": edge_relaxations,
        "max_frontier_size": max_frontier_size,
    }


# ------------------------------
# Example Driver
# ------------------------------

if __name__ == "__main__":
    import sys

    n, m, source, adj = load_graph(sys.argv[1])
    distances, metrics = range_sssp(n, source, adj)

    print("Dijkstra results")
    print("----------------")
    print(f"Vertices: {n}")
    print(f"Edges:    {m}")
    print(f"Source:   {source}")
    print()
    print(f"Runtime (s):           {metrics['runtime_sec']:.6f}")
    print(f"Edge relaxations:      {metrics['edge_relaxations']}")
    print(f"Max frontier size:     {metrics['max_frontier_size']}")
