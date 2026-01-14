"""
Pivot-Based, Ordering-Free SSSP (Practical Approximation)

This implementation is a practical, experimental realization of the ideas
presented in the poster on pivot-based single-source shortest paths (SSSP)
and the broader “breaking the sorting barrier” line of work.

Algorithmic perspective
-----------------------
Classical Dijkstra’s algorithm enforces a global total ordering over vertices
via a priority queue and extract-min operations. In contrast, this algorithm
deliberately avoids any form of global ordering or priority queue.

Instead, it maintains an active set of vertices whose outgoing edges may still
improve distance labels. In each iteration, a pivot vertex is selected from the
active set, defining a bounded distance region. Within this region, the
algorithm performs Bellman–Ford–style (label-correcting) relaxations until
local convergence. Vertices whose distances exceed the region bound are
deferred to future iterations.

Relation to the poster
----------------------
- Bellman–Ford–style relaxations:
  Yes. Label correction via repeated edge relaxation is the correctness
  backbone of the algorithm.

- Recursive or greedy Dijkstra:
  No. The algorithm does not use extract-min, does not finalize vertices
  greedily, and does not recursively invoke Dijkstra.

- Conceptual alignment:
  Conceptually aligned with the poster’s goals of avoiding global sorting,
  using pivot-based structure discovery, and progressively refining vertex
  sets via localized computation.

- Technical differences:
  This is a simplified, heuristic implementation intended for practical
  evaluation. It does not implement the full theoretical BMSSP framework,
  nor does it provide the same asymptotic guarantees.

Summary
-------
This implementation replaces Dijkstra’s global priority queue with a
pivot-based region selection strategy. For each pivot, Bellman–Ford–style
relaxations are applied within a bounded distance region, while vertices
outside the region are deferred. The algorithm iteratively refines the active
vertex set until all shortest-path distances converge, demonstrating a
structured, ordering-free approach to SSSP in practice.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import inf
from random import Random
import sys
from typing import Dict, Iterable, List, Tuple, Hashable, Optional, Set

from typing import List, Tuple, Dict
GraphAdj = List[List[Tuple[int, int]]]



import time
from collections import deque
from math import inf
from random import Random
from typing import Dict, Set


def pivot_based_sssp(
    adj: GraphAdj,
    source: int,
    *,
    radius: int | None = None,
    seed: int = 0,
    max_outer_iters: int = 10_000,
):
    """
    Pivot-based SSSP without global sorting.

    Returns:
        dist: list of distances
        metrics: dict with runtime and counters
    """

    n = len(adj)
    dist = [inf] * n
    dist[source] = 0

    rng = Random(seed)

    # --- Metrics ---
    edge_relaxations = 0
    max_frontier_size = 0

    # Active set = vertices that may still propagate improvements
    active: Set[int] = {source}

    # Heuristic radius if not provided
    if radius is None:
        radius = _estimate_radius(adj)

    start_time = time.perf_counter()

    outer = 0
    while active:
        outer += 1
        if outer > max_outer_iters:
            raise RuntimeError("Exceeded max_outer_iters")

        max_frontier_size = max(max_frontier_size, len(active))

        # ---- Pivot selection (unordered) ----
        pivot = rng.choice(tuple(active))
        threshold = dist[pivot] + radius

        # ---- Region processing ----
        queue = deque()
        in_region: Set[int] = set()
        deferred: Set[int] = set()

        # Seed region with all active vertices within threshold
        for v in active:
            if dist[v] <= threshold:
                queue.append(v)
                in_region.add(v)

        processed: Set[int] = set()

        while queue:
            u = queue.popleft()
            processed.add(u)

            du = dist[u]
            if du > threshold:
                deferred.add(u)
                continue

            for v, w in adj[u]:
                edge_relaxations += 1
                nd = du + w
                if nd < dist[v]:
                    dist[v] = nd
                    if nd <= threshold:
                        if v not in in_region:
                            in_region.add(v)
                            queue.append(v)
                        else:
                            queue.append(v)
                    else:
                        deferred.add(v)

        # ---- Active set refinement ----
        for u in processed:
            if dist[u] <= threshold:
                active.discard(u)

        active |= deferred

    runtime_sec = time.perf_counter() - start_time

    metrics = {
        "runtime_sec": runtime_sec,
        "edge_relaxations": edge_relaxations,
        "max_frontier_size": max_frontier_size,
    }

    return dist, metrics

def _estimate_radius(adj: GraphAdj) -> int:
    weights = []
    for outs in adj:
        for _, w in outs:
            if w > 0:
                weights.append(w)
            if len(weights) >= 1024:
                break
        if len(weights) >= 1024:
            break

    if not weights:
        return 1

    weights.sort()
    median = weights[len(weights) // 2]

    # Region size ≈ several typical edges
    return max(1, median * 8)

def load_graph(path: str):
    with open(path, "r", encoding="utf-8") as f:
        n, m, source = map(int, f.readline().split())
        adj = [[] for _ in range(n)]
        for line in f:
            if not line.strip():
                continue
            u, v, w = map(int, line.split())
            adj[u].append((v, w))
    return n, m, source, adj


# -------------------- Example usage --------------------
if __name__ == "__main__":
    n, m, source, adj = load_graph(sys.argv[1])

    dist, metrics = pivot_based_sssp(
        adj,
        source,
        radius=None,   # or set explicitly
        seed=0,
    )

    print("SSSP results")
    print("----------------")
    print(f"Vertices: {n}")
    print(f"Edges:    {m}")
    print(f"Source:   {source}")
    print()
    print(f"Runtime (s):           {metrics['runtime_sec']:.6f}")
    print(f"Edge relaxations:      {metrics['edge_relaxations']}")
    print(f"Max frontier size:     {metrics['max_frontier_size']}")
