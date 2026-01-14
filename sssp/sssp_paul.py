import heapq
import math
import time
from collections import defaultdict, deque
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
# Pivoted Range-based SSSP with bounded BF
# -------------------------------------------------

def pivoted_range_sssp(
    n: int,
    source: int,
    adj: GraphAdj,
    delta: int = None,
    bf_depth: int = None,
    pivot_threshold: int = None,
) -> Tuple[List[float], Dict[str, float]]:

    INF = float("inf")

    if delta is None:
        delta = max(64, int(math.log2(n) ** 2))

    if bf_depth is None:
        bf_depth = max(4, int(math.log2(n)))

    if pivot_threshold is None:
        pivot_threshold = bf_depth

    d = [INF] * n
    finished = [False] * n
    complete = [False] * n
    parent = [-1] * n

    d[source] = 0.0
    complete[source] = True

    buckets: Dict[int, List[int]] = defaultdict(list)
    buckets[0].append(source)

    edge_relaxations = 0
    max_frontier_size = 0
    bf_relaxations = 0

    start_time = time.perf_counter()
    current_band = 0

    while buckets:
        if current_band not in buckets:
            current_band = min(buckets.keys())

        band_vertices = buckets.pop(current_band)
        band_limit = (current_band + 1) * delta

        # --------------------------
        # Local Dijkstra (band)
        # --------------------------
        pq = []
        for u in band_vertices:
            if not finished[u]:
                heapq.heappush(pq, (d[u], u))

        max_frontier_size = max(max_frontier_size, len(pq))

        band_nodes = []
        while pq:
            du, u = heapq.heappop(pq)
            if finished[u]:
                continue
            if du >= band_limit:
                buckets[int(du // delta)].append(u)
                continue

            finished[u] = True
            band_nodes.append(u)

            for v, w in adj[u]:
                edge_relaxations += 1
                nd = du + w
                if nd < d[v]:
                    d[v] = nd
                    parent[v] = u
                    band = int(nd // delta)
                    buckets[band].append(v)
                    if band == current_band:
                        heapq.heappush(pq, (nd, v))

        if not band_nodes:
            current_band += 1
            continue

        # --------------------------
        # Build subtree sizes
        # --------------------------
        band_set = set(band_nodes)

        children = defaultdict(list)
        roots = []

        for u in band_nodes:
            p = parent[u]
            if p != -1 and p in band_set:
                children[p].append(u)
            else:
                roots.append(u)

        subtree_size = {}

        def dfs(u):
            size = 1
            for v in children[u]:
                size += dfs(v)
            subtree_size[u] = size
            return size

        for r in roots:
            dfs(r)


        pivots = {
            u for u in band_nodes
            if subtree_size[u] >= pivot_threshold
        }

        # --------------------------
        # Bounded Bellman-Ford
        # --------------------------
        bf_queue = deque()
        hop = {u: 0 for u in pivots}

        for u in pivots:
            bf_queue.append(u)
            complete[u] = True

        while bf_queue:
            u = bf_queue.popleft()
            if hop[u] >= bf_depth:
                continue
            for v, w in adj[u]:
                bf_relaxations += 1
                if d[u] + w < d[v]:
                    d[v] = d[u] + w
                    parent[v] = u
                    hop[v] = hop[u] + 1
                    complete[v] = True
                    bf_queue.append(v)

        # --------------------------
        # Reinsert incomplete nodes
        # --------------------------
        for u in band_nodes:
            if not complete[u]:
                buckets[int(d[u] // delta)].append(u)

        current_band += 1

    runtime_sec = time.perf_counter() - start_time

    return d, {
        "runtime_sec": runtime_sec,
        "edge_relaxations": edge_relaxations,
        "bf_relaxations": bf_relaxations,
        "max_frontier_size": max_frontier_size,
        "bf_depth": bf_depth,
        "pivot_threshold": pivot_threshold,
        "delta": delta,
    }


# ------------------------------
# Example Driver
# ------------------------------

if __name__ == "__main__":
    import sys

    n, m, source, adj = load_graph(sys.argv[1])

    distances, metrics = pivoted_range_sssp(
        n,
        source,
        adj,
    )

    print("Pivoted Range SSSP Results")
    print("---------------------------")
    print(f"Vertices: {n}")
    print(f"Edges:    {m}")
    print(f"Source:   {source}")
    print()
    for k, v in metrics.items():
        print(f"{k:20s}: {v}")
