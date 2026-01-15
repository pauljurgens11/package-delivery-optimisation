#!/usr/bin/env python3
"""
Compare performance of BMSSP, Dijkstra, and Bellman–Ford on generated graphs.

This script scans ``generator/generated-graphs/*.txt``, runs all three SSSP
implementations on each instance, and writes a CSV with comparable metrics.
Optionally, if ``matplotlib`` is installed, it also produces simple PNG plots.
"""

from __future__ import annotations

import csv
import glob
import os
import time
from pathlib import Path
from typing import Dict, List

from generator.graph_generator import load_edge_list_txt
from sssp.ssspx.baselines import BellmanFordSolver, DijkstraSolver
from sssp.ssspx.graph import Graph
from sssp.ssspx.logger import NoopLogger
from sssp.ssspx.solver import SolverConfig, SSSPSolver


def _run_solver(name: str, G: Graph, source: int) -> Dict[str, object]:
    cfg = SolverConfig(use_transform=(name == "bmssp"), frontier="block")
    logger = NoopLogger()
    SolverCls = {"bmssp": SSSPSolver, "dijkstra": DijkstraSolver, "bellman-ford": BellmanFordSolver}[name]
    # Track peak memory via tracemalloc for fair comparison
    import tracemalloc

    tracemalloc.start()
    t0 = time.perf_counter()
    solver = SolverCls(G, source, config=cfg, logger=logger)  # type: ignore[arg-type]
    res = solver.solve()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    _cur, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mib = peak_bytes / (1024 * 1024)

    metrics = solver.metrics(wall_ms=wall_ms, peak_mib=peak_mib)
    row: Dict[str, object] = {
        "algo": name,
        "wall_ms": metrics.wall_ms,
        "peak_mib": metrics.peak_mib,
        "edges_relaxed": metrics.counters.get("edges_relaxed", 0),
    }
    # keep full metrics for later digging if desired
    row.update({f"counter_{k}": v for k, v in metrics.counters.items()})
    # basic correctness sanity: ensure distances list is present
    row["num_vertices"] = len(res.distances)
    return row


def main() -> None:
    base_dir = Path("generator/generated-graphs")
    out_csv = Path("algorithm_comparison.csv")

    files = sorted(glob.glob(str(base_dir / "*.txt")))
    if not files:
        raise SystemExit(f"No .txt graphs found under {base_dir}")

    fieldnames: List[str] = [
        "file",
        "test_set",
        "n",
        "m",
        "source",
        "algo",
        "wall_ms",
        "peak_mib",
        "edges_relaxed",
        # Optional extra scalar metrics (present for some algorithms)
        "max_frontier_size",
        "iterations",
    ]

    rows: List[Dict[str, object]] = []
    for path in files:
        gen = load_edge_list_txt(path)
        G = Graph.from_edges(gen.n, gen.edges)

        # derive simple test-set name, e.g. "A_baseline_scaling"
        fname = os.path.basename(path)
        test_set = fname.split("_n", 1)[0]

        for algo in ("bmssp", "dijkstra", "bellman-ford"):
            r = _run_solver(algo, G, gen.source)
            row: Dict[str, object] = {
                "file": fname,
                "test_set": test_set,
                "n": gen.n,
                "m": gen.m,
                "source": gen.source,
            }
            row.update(r)
            # Surface a couple of key counters as top-level CSV columns when present
            if "counter_max_frontier_size" in r:
                row["max_frontier_size"] = r["counter_max_frontier_size"]
            if "counter_iterations" in r:
                row["iterations"] = r["counter_iterations"]
            rows.append(row)
            print(
                f"{fname:60s} algo={algo:12s} n={gen.n:7d} m={gen.m:8d} "
                f"wall_ms={row['wall_ms']:.2f}"
            )

    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"\nWrote CSV summary to {out_csv}")


if __name__ == "__main__":
    main()


