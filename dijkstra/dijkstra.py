#!/usr/bin/env python3
"""
Dijkstra's algorithm for SSSP benchmarking.

Input format (edge list):
    n m source
    u v w
    u v w
    ...

Metrics collected:
- Total runtime
- Number of edge relaxations
- Maximum frontier size (priority queue size)
"""

import time
import heapq
import sys
from typing import List, Tuple


GraphAdj = List[List[Tuple[int, int]]]


def load_graph(path: str) -> Tuple[int, int, int, GraphAdj]:
    """
    Load graph from edge-list text file.
    """
    with open(path, "r", encoding="utf-8") as f:
        n, m, source = map(int, f.readline().split())
        adj: GraphAdj = [[] for _ in range(n)]
        for line in f:
            if not line.strip():
                continue
            u, v, w = map(int, line.split())
            adj[u].append((v, w))
    return n, m, source, adj


def dijkstra(adj: GraphAdj, source: int):
    """
    Standard Dijkstra with a binary heap.

    Returns:
        dist: list of distances
        metrics: dict with runtime, relaxations, max_frontier
    """
    n = len(adj)
    INF = float("inf")
    dist = [INF] * n
    dist[source] = 0

    pq = [(0, source)]
    relaxations = 0
    max_frontier = 1

    start = time.perf_counter()

    while pq:
        max_frontier = max(max_frontier, len(pq))
        d_u, u = heapq.heappop(pq)

        # Lazy deletion
        if d_u != dist[u]:
            continue

        for v, w in adj[u]:
            relaxations += 1
            nd = d_u + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    end = time.perf_counter()

    metrics = {
        "runtime_sec": end - start,
        "edge_relaxations": relaxations,
        "max_frontier_size": max_frontier,
    }

    return dist, metrics


def main():
    if len(sys.argv) != 2:
        print("Usage: dijkstra.py <graph_file>")
        sys.exit(1)

    graph_file = sys.argv[1]
    n, m, source, adj = load_graph(graph_file)

    dist, metrics = dijkstra(adj, source)

    print("Dijkstra results")
    print("----------------")
    print(f"Vertices: {n}")
    print(f"Edges:    {m}")
    print(f"Source:   {source}")
    print()
    print(f"Runtime (s):           {metrics['runtime_sec']:.6f}")
    print(f"Edge relaxations:      {metrics['edge_relaxations']}")
    print(f"Max frontier size:     {metrics['max_frontier_size']}")


if __name__ == "__main__":
    main()
