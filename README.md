# package-delivery-optimisation

## Problem formulation

The goal of the project is to implement and evaluate multiple algorithms for the single-source shortest path (SSSP) problem on directed graphs, specifically:

- Dijkstra’s algorithm
- Bellman–Ford algorithm
- The algorithm proposed in "Breaking the Sorting Barrier for Directed Single-Source Shortest Paths" (combination of the 2)

The primary research objectives are:
- Does the proposed algorithm outperform classical approaches in practice? Why?
- Under which conditions does this happen?

## Scope & Constraints

To keep the project feasible, we will:

- Focus on directed graphs with non-negative edge weights
- Single-source shortest paths
- In-memory graphs
- Sequential implementations

This ensures fair comparison and avoids complicating factors outside the paper’s scope.

## Algorithms to implement

### Baseline Algorithms

These serve as reference points.

Dijkstra’s Algorithm:

- Expected complexity: Θ((n + m) log n).
- Works by prioritising which paths to explore. Instead of exploring all possible paths equally, it favours lower-cost paths.
- You can assign lower costs to encourage moving on roads, higher costs to avoid forests or hills, even higher costs to avoid going near enemy fortifications, and more.

Bellman–Ford Algorithm:

- Complexity: O(mn) (or O(mk) for k iterations).
- Solves the same problem as Dijkstra's but with a non-greedy approach.
- It's less efficient but more versatile and can be used for graphs with negative edge cost where some other algorithms won't work.

### Algorithm from the Paper

Implement the algorithm described in the paper:

- Recursive / divide-and-conquer structure
- Combines the strengths of both approaches while mitigating their weaknesses.
- Instead of maintaining a total order over all frontier vertices as in Dijkstra’s algorithm, it works with a carefully controlled frontier whose size is kept small. When the frontier grows too large, the algorithm temporarily switches to a limited number of Bellman–Ford–style relaxation steps to resolve many vertices at once and to identify a smaller set of "pivot" vertices that truly matter.

## Implementation Details

Algorithms are implemented in Python. They will take an in-memory graph as input.

## Metrics and Comparisons

Metrics:

- Total runtime
- Number of edge relaxations
- Frontier size over time (for the paper algorithm)
- Memory usage

## Testing Plan in Detail

The algorithms are evaluated on a small set of representative graph families: random graphs, directed acyclic graphs, grid graphs, and preferential-attachment graphs. These cover average-case behavior, propagation-friendly instances, worst-case frontier growth, and realistic skewed-degree structures. Graph size, density, and weight distributions are varied one parameter at a time to isolate their impact on performance.

This selection is sufficient to expose the key algorithmic trade-offs without unnecessary redundancy.

The graph generator can be used to generated these graphs:

| Test set | Graph family (generator)                     | What it represents                          | Sizes `n`                                       | Density / edges `m`                          | Weight distribution | # instances (seeds) |
| -------- | -------------------------------------------- | ------------------------------------------- | ----------------------------------------------- | -------------------------------------------- | ------------------- | ------------------- |
| A        | Random directed (Erdős–Rényi, `erdos_renyi`) | Average-case baseline                       | 1e3, 1e4, 1e5 (scaling test)                                   | Sparse: `m = 4n`                             | `uniform`           | 3                   |
| B        | Random directed (Erdős–Rényi, `erdos_renyi`) | Sorting stress via many equal-ish distances | 1e5                                   | Sparse: `m = 4n`                             | `small_int`         | 3                   |
| C        | Random directed (Erdős–Rényi, `erdos_renyi`) | Effect of density (frontier growth)         | 1e5                                   | Medium: `m = 16n`                            | `uniform`           | 3                   |
| D        | DAG (`dag`)                                  | Propagation-friendly / limited-depth paths  | 1e5                                   | Sparse: `m = 4n`                             | `uniform`           | 3                   |
| E        | Grid (`grid`)                                | Structured worst-case frontier growth       | ≈316×316 (~1e5) | Natural grid edges (plus none)               | `uniform`           | 3                   |
| F        | Preferential attachment (`barabasi_albert`)  | Skewed-degree “realistic” networks (hubs) (fat tails)   | 1e5                                   | Medium via attachment (≈`m ≈ n * ba_attach`) | `log_uniform`       | 3                   |
