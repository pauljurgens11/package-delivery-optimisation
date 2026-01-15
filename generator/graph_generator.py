#!/usr/bin/env python3
"""
Directed weighted graph generator for SSSP algorithm evaluation.

SUPPORTED GRAPH TYPES
---------------------
1. erdos_renyi
   Random directed graphs with uniformly sampled edges.
   Use for:
     - Baseline / average-case performance
     - Scalability tests with increasing n and m

2. dag
   Directed acyclic graphs (edges only from lower- to higher-index vertices).
   Use for:
     - Evaluating Bellman–Ford-style propagation
     - Testing performance when shortest paths have limited depth

3. grid
   2D grid graphs with edges between neighboring vertices (bidirectional).
   Use for:
     - Structured graphs with many equal-length shortest paths
     - Stress-testing frontier growth and ordering overhead

4. barabasi_albert
   Preferential-attachment graphs with hub-heavy degree distributions.
   Use for:
     - Realistic, skewed-degree graphs
     - Evaluating pivot selection and frontier reduction effectiveness

WEIGHT DISTRIBUTIONS
--------------------
- uniform: evenly distributed edge weights
- small_int: many equal or similar weights (stresses sorting)
- log_uniform / exp: heavy-tailed distributions

RECOMMENDED FOR EXPERIMENTS
---------------------------
A representative evaluation should include:
  - erdos_renyi + uniform weights (baseline)
  - dag + uniform weights (propagation-friendly)
  - grid + uniform or small_int weights (frontier stress)
  - barabasi_albert + log_uniform weights (realistic skew)

All generated graphs use non-negative weights and are safe for Dijkstra.
The output format is compatible with Dijkstra, Bellman–Ford,
and sorting-barrier-breaking SSSP algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal
import random
import math

try:
    # When imported as part of the ``generator`` package
    from .test_cases import ALL_TEST_SETS  # type: ignore[import]
except ImportError:  # pragma: no cover - fallback for script-style execution
    # When run as a standalone script from inside the ``generator/`` directory
    from test_cases import ALL_TEST_SETS  # type: ignore[import]

GraphAdj = List[List[Tuple[int, int]]]
EdgeList = List[Tuple[int, int, int]]

WeightDist = Literal["uniform", "small_int", "log_uniform", "exp"]
GraphType = Literal["erdos_renyi", "dag", "grid", "barabasi_albert"]


@dataclass(frozen=True)
class GeneratedGraph:
    n: int
    m: int
    adj: GraphAdj
    edges: EdgeList
    source: int
    metadata: dict


def _sample_weight(
    rng: random.Random,
    dist: WeightDist,
    w_min: int,
    w_max: int,
) -> int:
    if w_min < 0:
        raise ValueError("w_min must be >= 0 for Dijkstra-safe graphs.")
    if w_max < w_min:
        raise ValueError("w_max must be >= w_min.")

    if dist == "uniform":
        return rng.randint(w_min, w_max)

    if dist == "small_int":
        # Concentrate weights in a small range (useful for stress-testing bucket-like behaviors)
        hi = min(w_max, w_min + 10)
        return rng.randint(w_min, hi)

    if dist == "log_uniform":
        # Sample uniformly in log-space, then round to int (non-negative).
        # Avoid log(0) by shifting.
        a = max(1, w_min + 1)
        b = max(a, w_max + 1)
        x = math.exp(rng.uniform(math.log(a), math.log(b)))
        return max(w_min, min(w_max, int(round(x - 1))))

    if dist == "exp":
        # Exponential-like: many small weights, occasional larger ones.
        # Scale to [w_min, w_max].
        if w_max == w_min:
            return w_min
        lam = 1.0 / max(1.0, (w_max - w_min) / 4.0)
        x = rng.expovariate(lam)
        w = int(w_min + min(w_max - w_min, round(x)))
        return w

    raise ValueError(f"Unknown weight distribution: {dist}")


def _build_from_edges(n: int, edges: EdgeList) -> Tuple[GraphAdj, EdgeList]:
    adj: GraphAdj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
    return adj, edges


def generate_graph(
    *,
    n: int,
    m: Optional[int] = None,
    graph_type: GraphType = "erdos_renyi",
    weight_dist: WeightDist = "uniform",
    w_min: int = 1,
    w_max: int = 100,
    seed: Optional[int] = 0,
    source: int = 0,
    allow_self_loops: bool = False,
    ensure_weakly_connected: bool = True,
    # Parameters for specific families:
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
    ba_m0: int = 5,        # initial clique size (>=2)
    ba_attach: int = 3,    # edges per new node (>=1)
) -> GeneratedGraph:
    """
    Generate a directed weighted graph.

    Notes:
    - If ensure_weakly_connected=True, we add a simple backbone chain (i->i+1) to avoid
      totally disconnected instances. This preserves directedness but ensures weak connectivity.
    - For DAG graphs, edges go from lower index to higher index (acyclic).

    Returns:
        GeneratedGraph with adj list and edge list.
    """
    if n <= 0:
        raise ValueError("n must be > 0.")
    if not (0 <= source < n):
        raise ValueError("source must be in [0, n).")

    rng = random.Random(seed)

    # Decide m defaults per family
    if graph_type == "grid":
        if grid_rows is None or grid_cols is None:
            # Choose a near-square grid
            grid_rows = int(math.isqrt(n))
            grid_rows = max(1, grid_rows)
            grid_cols = max(1, (n + grid_rows - 1) // grid_rows)
        if grid_rows * grid_cols < n:
            raise ValueError("grid_rows*grid_cols must be >= n.")
        # Grid base edges (right/down) as directed both ways by default (stronger connectivity)
        # We'll compute m based on actual construction.
    else:
        if m is None:
            # Reasonable default: sparse-ish
            m = min(n * 4, n * (n - 1))
    if m is not None and m < 0:
        raise ValueError("m must be >= 0.")

    edges_set: set[Tuple[int, int]] = set()
    edges: EdgeList = []

    def add_edge(u: int, v: int) -> None:
        if not allow_self_loops and u == v:
            return
        key = (u, v)
        if key in edges_set:
            return
        w = _sample_weight(rng, weight_dist, w_min, w_max)
        edges_set.add(key)
        edges.append((u, v, w))

    # Optional backbone to avoid pathological disconnected graphs
    if ensure_weakly_connected and n >= 2 and graph_type != "grid":
        for i in range(n - 1):
            add_edge(i, i + 1)

    if graph_type == "erdos_renyi":
        # Random directed edges until reaching target m
        target_m = min(m if m is not None else 0, n * n if allow_self_loops else n * (n - 1))
        # Account for any edges already added (backbone)
        while len(edges) < target_m:
            u = rng.randrange(n)
            v = rng.randrange(n)
            add_edge(u, v)

    elif graph_type == "dag":
        # Directed acyclic: u < v
        target_m = min(m if m is not None else 0, n * (n - 1) // 2)
        while len(edges) < target_m:
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            add_edge(u, v)

    elif graph_type == "grid":
        # Build a directed grid on first n cells in row-major order.
        R = grid_rows
        C = grid_cols
        idx = lambda r, c: r * C + c

        def in_range(x: int) -> bool:
            return 0 <= x < n

        # Add edges between neighbors (both directions) to create a challenging but structured instance
        for r in range(R):
            for c in range(C):
                u = idx(r, c)
                if not in_range(u):
                    continue
                # right neighbor
                if c + 1 < C:
                    v = idx(r, c + 1)
                    if in_range(v):
                        add_edge(u, v)
                        add_edge(v, u)
                # down neighbor
                if r + 1 < R:
                    v = idx(r + 1, c)
                    if in_range(v):
                        add_edge(u, v)
                        add_edge(v, u)

        # Optionally add extra random edges to reach m (if provided)
        if m is not None:
            target_m = min(m, n * n if allow_self_loops else n * (n - 1))
            while len(edges) < target_m:
                u = rng.randrange(n)
                v = rng.randrange(n)
                add_edge(u, v)

    elif graph_type == "barabasi_albert":
        # Fast preferential attachment using a node pool approximation.
        # This avoids O(n^2) behavior and is sufficient for benchmarking purposes.

        m0 = max(2, min(ba_m0, n))
        attach = max(1, min(ba_attach, m0))

        node_pool = []

        def add_edge_ba(u: int, v: int) -> None:
            if not allow_self_loops and u == v:
                return
            key = (u, v)
            if key in edges_set:
                return
            w = _sample_weight(rng, weight_dist, w_min, w_max)
            edges_set.add(key)
            edges.append((u, v, w))
            node_pool.append(u)
            node_pool.append(v)

        # Initial clique
        for u in range(m0):
            for v in range(m0):
                if u != v:
                    add_edge_ba(u, v)

        # Grow graph
        for new in range(m0, n):
            chosen = set()
            while len(chosen) < attach:
                chosen.add(rng.choice(node_pool))

            for old in chosen:
                add_edge_ba(new, old)
                if rng.random() < 0.5:
                    add_edge_ba(old, new)

    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    adj, edges_out = _build_from_edges(n, edges)
    return GeneratedGraph(
        n=n,
        m=len(edges_out),
        adj=adj,
        edges=edges_out,
        source=source,
        metadata={
            "graph_type": graph_type,
            "weight_dist": weight_dist,
            "w_min": w_min,
            "w_max": w_max,
            "seed": seed,
            "ensure_weakly_connected": ensure_weakly_connected,
            "allow_self_loops": allow_self_loops,
        },
    )


def save_edge_list_txt(g: GeneratedGraph, path: str) -> None:
    """
    Save as a simple text format:
        n m source
        u v w
        u v w
        ...
    This is easy to parse in any language.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{g.n} {g.m} {g.source}\n")
        for u, v, w in g.edges:
            f.write(f"{u} {v} {w}\n")


def load_edge_list_txt(path: str) -> GeneratedGraph:
    """
    Load the format produced by save_edge_list_txt.
    """
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        n, m, source = int(header[0]), int(header[1]), int(header[2])
        edges: EdgeList = []
        for line in f:
            if not line.strip():
                continue
            u, v, w = map(int, line.split())
            edges.append((u, v, w))
    adj, edges_out = _build_from_edges(n, edges)
    return GeneratedGraph(
        n=n,
        m=len(edges_out),
        adj=adj,
        edges=edges_out,
        source=source,
        metadata={"loaded_from": path},
    )


import os

if __name__ == "__main__":
    os.makedirs("generated-graphs", exist_ok=True)

    for name, cases in ALL_TEST_SETS.items():
        for i, cfg in enumerate(cases):
            graph = generate_graph(**cfg)
            seed = cfg.get("seed", "na")
            n = cfg["n"]
            out = f"generated-graphs/{name}_n{n}_seed{seed}.txt"
            save_edge_list_txt(graph, out)
            print(f"Saved {out} (n={graph.n}, m={graph.m})")

    # Example usage (modify as needed):
    # g = generate_graph(
    #     n=10_000,
    #     m=40_000,
    #     graph_type="erdos_renyi",
    #     weight_dist="uniform",
    #     w_min=1,
    #     w_max=1000,
    #     seed=42,
    #     source=0,
    #     ensure_weakly_connected=True,
    # )

