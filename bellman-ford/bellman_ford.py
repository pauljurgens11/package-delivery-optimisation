#!/usr/bin/env python3
"""
Bellman–Ford algorithm for SSSP benchmarking.

Input format (edge list):
    n m source
    u v w
    u v w
    ...

Metrics collected:
- Total runtime
- Number of edge relaxations
- Number of iterations completed
"""

import time
import sys
from typing import List, Tuple


EdgeList = List[Tuple[int, int, int]]


def load_graph(path: str) -> Tuple[int, int, int, EdgeList]:
    """
    Load graph from edge-list text file.
    """
    with open(path, "r", encoding="utf-8") as f:
        n, m, source = map(int, f.readline().split())
        edges: EdgeList = []
        for line in f:
            if not line.strip():
                continue
            u, v, w = map(int, line.split())
            edges.append((u, v, w))
    return n, m, source, edges


def bellman_ford(n: int, edges: EdgeList, source: int, max_iters: int | None = None):
    """
    Standard Bellman–Ford with early stopping.

    Args:
        n: number of vertices
        edges: edge list (u, v, w)
        source: source vertex
        max_iters: optional cap on iterations (k in O(mk))

    Returns:
        dist: list of distances
        metrics: dict with runtime, relaxations, iterations
    """
    INF = float("inf")
    dist = [INF] * n
    dist[source] = 0

    relaxations = 0
    iterations = 0

    start = time.perf_counter()

    # Run up to n-1 iterations unless capped
    limit = n - 1 if max_iters is None else min(max_iters, n - 1)

    for i in range(limit):
        updated = False
        iterations += 1

        for u, v, w in edges:
            relaxations += 1
            if dist[u] != INF:
                nd = dist[u] + w
                if nd < dist[v]:
                    dist[v] = nd
                    updated = True

        if not updated:
            break

    end = time.perf_counter()

    metrics = {
        "runtime_sec": end - start,
        "edge_relaxations": relaxations,
        "iterations": iterations,
    }

    return dist, metrics


def main():
    if len(sys.argv) != 2:
        print("Usage: bellman_ford.py <graph_file>")
        sys.exit(1)

    graph_file = sys.argv[1]
    n, m, source, edges = load_graph(graph_file)

    dist, metrics = bellman_ford(n, edges, source)

    print("Bellman–Ford results")
    print("--------------------")
    print(f"Vertices: {n}")
    print(f"Edges:    {m}")
    print(f"Source:   {source}")
    print()
    print(f"Runtime (s):           {metrics['runtime_sec']:.6f}")
    print(f"Edge relaxations:      {metrics['edge_relaxations']}")
    print(f"Iterations completed: {metrics['iterations']}")


if __name__ == "__main__":
    main()
