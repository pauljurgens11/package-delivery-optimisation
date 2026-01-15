from __future__ import annotations

import math
import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional

from .graph import Float, Graph, Vertex
from .logger import Logger, NoopLogger
from .solver import SSSPResult, SolverConfig, SolverMetrics


@dataclass
class _BaseBaselineSolver:
    """Common pieces shared between baseline solvers."""

    G: Graph
    source: Vertex
    config: Optional[SolverConfig] = None
    logger: Optional[Logger] = None

    def __post_init__(self) -> None:
        if not (0 <= self.source < self.G.n):
            raise ValueError("source must be a valid vertex id.")
        self.cfg = self.config or SolverConfig(use_transform=False)
        self.logger = self.logger or NoopLogger()
        self._dist: List[Float] = [math.inf] * self.G.n
        self._pred: List[Optional[Vertex]] = [None] * self.G.n
        # Counters aligned with SSSPSolver.summary where possible
        self.counters: Dict[str, int] = {
            "edges_relaxed": 0,
            "pulls": 0,
            "findpivots_rounds": 0,
            "basecase_pops": 0,
            "iterations_protected": 0,
        }


    def summary(self) -> Dict[str, int]:
        return dict(self.counters)

    def metrics(self, wall_ms: float, peak_mib: float | None = None) -> SolverMetrics:
        m = sum(len(lst) for lst in self.G.adj)
        return SolverMetrics(
            n=self.G.n,
            m=m,
            frontier="heap",
            transform=False,
            counters=self.summary(),
            wall_ms=wall_ms,
            peak_mib=peak_mib,
        )


class DijkstraSolver(_BaseBaselineSolver):
    def __init__(
        self,
        G: Graph,
        source: Vertex,
        config: Optional[SolverConfig] = None,
        logger: Logger | None = None,
        sources: Optional[List[Vertex]] = None,  # ignored, kept for CLI compatibility
    ) -> None:
        super().__init__(G=G, source=source, config=config, logger=logger)

    def solve(self) -> SSSPResult:
        n = self.G.n
        dist = self._dist
        pred = self._pred
        dist[self.source] = 0.0

        pq: List[tuple[Float, Vertex]] = [(0.0, self.source)]
        max_frontier = 1

        while pq:
            max_frontier = max(max_frontier, len(pq))
            d_u, u = heapq.heappop(pq)
            # lazy deletion
            if d_u != dist[u]:
                continue
            for v, w in self.G.adj[u]:
                self.counters["edges_relaxed"] += 1
                nd = d_u + w
                if nd < dist[v]:
                    dist[v] = nd
                    pred[v] = u
                    heapq.heappush(pq, (nd, v))

        # Record frontier usage as an extra counter for easier comparison
        self.counters.setdefault("max_frontier_size", max_frontier)
        return SSSPResult(distances=dist, predecessors=pred)


class BellmanFordSolver(_BaseBaselineSolver):
    def __init__(
        self,
        G: Graph,
        source: Vertex,
        config: Optional[SolverConfig] = None,
        logger: Logger | None = None,
        sources: Optional[List[Vertex]] = None,  # ignored, kept for CLI compatibility
        max_iters: Optional[int] = None,
    ) -> None:
        super().__init__(G=G, source=source, config=config, logger=logger)
        # Up to n-1 iterations unless capped
        self._max_iters = max_iters

    def solve(self) -> SSSPResult:
        n = self.G.n
        dist = self._dist
        pred = self._pred
        dist[self.source] = 0.0

        iterations = 0
        limit = n - 1 if self._max_iters is None else min(self._max_iters, n - 1)

        for _ in range(limit):
            updated = False
            iterations += 1
            for u in range(n):
                du = dist[u]
                if du == math.inf:
                    continue
                for v, w in self.G.adj[u]:
                    self.counters["edges_relaxed"] += 1
                    nd = du + w
                    if nd < dist[v]:
                        dist[v] = nd
                        pred[v] = u
                        updated = True
            if not updated:
                break

        # Expose iterations in counters for downstream analysis
        self.counters.setdefault("iterations", iterations)
        return SSSPResult(distances=dist, predecessors=pred)


