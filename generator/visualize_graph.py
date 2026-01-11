#!/usr/bin/env python3
"""
Graph visualization utility for GeneratedGraph / edge-list files.

Features:
- Loads graphs saved via save_edge_list_txt(...)
- Visualizes directed, weighted graphs
- Automatically downsamples large graphs for readability
- Highlights the source node

Example usage:

```
python visualize_graph.py graph.txt
```
OR
```
python visualize_graph.py graph.txt \
    --max-edges 200 \
    --layout kamada_kawai \
    --show-weights
```
"""

from __future__ import annotations

import argparse
import random
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import networkx as nx


def load_edge_list_txt(path: str) -> Tuple[int, int, int, list[tuple[int, int, int]]]:
    """
    Load graph from the text format:
        n m source
        u v w
        u v w
        ...
    """
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().split()
        n, m, source = map(int, header)
        edges = []
        for line in f:
            if not line.strip():
                continue
            u, v, w = map(int, line.split())
            edges.append((u, v, w))
    return n, m, source, edges


def downsample_edges(
    edges: Iterable[tuple[int, int, int]],
    max_edges: int,
    seed: int = 0,
) -> list[tuple[int, int, int]]:
    """
    Randomly sample edges if the graph is too large to visualize.
    """
    edges = list(edges)
    if len(edges) <= max_edges:
        return edges
    rng = random.Random(seed)
    return rng.sample(edges, max_edges)


def visualize_graph(
    n: int,
    source: int,
    edges: list[tuple[int, int, int]],
    *,
    layout: str = "spring",
    show_weights: bool = False,
    node_size: int = 300,
):
    """
    Render the graph using NetworkX + Matplotlib.
    """

    G = nx.DiGraph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    plt.figure(figsize=(12, 10))

    node_colors = [
        "tab:red" if node == source else "tab:blue"
        for node in G.nodes
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_size,
        alpha=0.9,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="->",
        arrowsize=12,
        width=1.2,
        alpha=0.6,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8,
        font_color="black",
    )

    if show_weights:
        edge_labels = {
            (u, v): w for u, v, w in edges
        }
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=7,
        )

    plt.title("Directed Weighted Graph Visualization", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize a generated graph")
    parser.add_argument("path", help="Path to graph.txt")
    parser.add_argument("--max-edges", type=int, default=300,
                        help="Maximum edges to display (sampling if larger)")
    parser.add_argument("--layout", choices=["spring", "kamada_kawai", "shell"],
                        default="spring")
    parser.add_argument("--show-weights", action="store_true",
                        help="Render edge weights (recommended only for very small graphs)")
    parser.add_argument("--node-size", type=int, default=300)

    args = parser.parse_args()

    n, m, source, edges = load_edge_list_txt(args.path)
    edges = downsample_edges(edges, args.max_edges)

    visualize_graph(
        n=n,
        source=source,
        edges=edges,
        layout=args.layout,
        show_weights=args.show_weights,
        node_size=args.node_size,
    )


if __name__ == "__main__":
    main()
