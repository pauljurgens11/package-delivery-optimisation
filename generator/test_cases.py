"""
Test case definitions for SSSP algorithm evaluation.

This module defines a small, representative set of graph-generation
configurations used to evaluate:

- Dijkstra’s algorithm
- Bellman–Ford algorithm
- The sorting-barrier-breaking SSSP algorithm

The test sets are intentionally limited in number but cover the most
important structural regimes:
  - average-case random graphs
  - sorting stress via near-equal distances
  - density-induced frontier growth
  - propagation-friendly DAGs
  - structured worst-case frontiers (grids)
  - skewed-degree, hub-heavy graphs

Each test case is a dictionary of keyword arguments compatible with
`generate_graph(...)`.
"""

# Main graph size for non-scaling tests
N_MAIN = 100_000

# Common settings applied to all tests
COMMON = dict(
    source=0,
    ensure_weakly_connected=True,
    allow_self_loops=False,
)

# -------------------------------------------------------------------
# A — Baseline scaling (Erdős–Rényi, sparse, uniform weights)
# -------------------------------------------------------------------

A_SIZES = [1_000, 10_000, 100_000]
A_SEEDS = [0, 1, 2]

TEST_A_BASELINE_SCALING = [
    dict(
        n=n,
        m=4 * n,
        graph_type="erdos_renyi",
        weight_dist="uniform",
        w_min=1,
        w_max=1_000,
        seed=seed,
        **COMMON,
    )
    for n in A_SIZES
    for seed in A_SEEDS
]

# -------------------------------------------------------------------
# B — Sorting stress (Erdős–Rényi, sparse, small integer weights)
# -------------------------------------------------------------------

B_SEEDS = [0, 1, 2]

TEST_B_SORTING_STRESS = [
    dict(
        n=N_MAIN,
        m=4 * N_MAIN,
        graph_type="erdos_renyi",
        weight_dist="small_int",
        w_min=1,
        w_max=1_000,
        seed=seed,
        **COMMON,
    )
    for seed in B_SEEDS
]

# -------------------------------------------------------------------
# C — Density impact (Erdős–Rényi, medium density)
# -------------------------------------------------------------------

C_SEEDS = [0, 1, 2]

TEST_C_DENSITY_IMPACT = [
    dict(
        n=N_MAIN,
        m=16 * N_MAIN,
        graph_type="erdos_renyi",
        weight_dist="uniform",
        w_min=1,
        w_max=1_000,
        seed=seed,
        **COMMON,
    )
    for seed in C_SEEDS
]

# -------------------------------------------------------------------
# D — Propagation-friendly structure (DAG)
# -------------------------------------------------------------------

D_SEEDS = [0, 1, 2]

TEST_D_DAG = [
    dict(
        n=N_MAIN,
        m=4 * N_MAIN,
        graph_type="dag",
        weight_dist="uniform",
        w_min=1,
        w_max=1_000,
        seed=seed,
        **COMMON,
    )
    for seed in D_SEEDS
]

# -------------------------------------------------------------------
# E — Structured frontier stress (Grid)
# -------------------------------------------------------------------

E_SEEDS = [0, 1, 2]
GRID_ROWS = 316
GRID_COLS = 317  # GRID_ROWS * GRID_COLS >= 100_000

TEST_E_GRID = [
    dict(
        n=N_MAIN,
        m=None,  # use natural grid edges only
        graph_type="grid",
        grid_rows=GRID_ROWS,
        grid_cols=GRID_COLS,
        weight_dist="uniform",
        w_min=1,
        w_max=10,   # small weights to increase ties
        seed=seed,
        **COMMON,
    )
    for seed in E_SEEDS
]

# -------------------------------------------------------------------
# F — Skewed-degree realism (Barabási–Albert)
# -------------------------------------------------------------------

F_SEEDS = [0, 1, 2]

TEST_F_PREFERENTIAL_ATTACHMENT = [
    dict(
        n=N_MAIN,
        m=None,               # let BA control density
        graph_type="barabasi_albert",
        ba_m0=10,
        ba_attach=3,
        weight_dist="log_uniform",
        w_min=1,
        w_max=10_000,
        seed=seed,
        **COMMON,
    )
    for seed in F_SEEDS
]

# -------------------------------------------------------------------
# Convenience collections
# -------------------------------------------------------------------

ALL_TEST_SETS = {
    # "A_baseline_scaling": TEST_A_BASELINE_SCALING,
    # "B_sorting_stress": TEST_B_SORTING_STRESS,
    # "C_density_impact": TEST_C_DENSITY_IMPACT,
    # "D_dag": TEST_D_DAG,
    # "E_grid": TEST_E_GRID,
    "F_preferential_attachment": TEST_F_PREFERENTIAL_ATTACHMENT,
}

# Flattened list (useful for simple runners)
ALL_TEST_CASES = [
    case
    for cases in ALL_TEST_SETS.values()
    for case in cases
]
