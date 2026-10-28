"""Exact full-ambient-space minimum spanning tree rounding for EVA factors.

Build the MST of the COMPLETE Euclidean point graph whose vertices are
the unprojected rows of Y. Distances may be L1, L2, or L-infinity.
Remove exactly the longest tree edge; the two connected components
form a binary Max-Cut candidate on the ORIGINAL weighted input graph.

Complexity: O(n^2 p) distance arithmetic and O(n p) memory (besides the
input factor), using an optional compiled Numba Prim kernel. No PCA,
no k-nearest-neighbor approximation, no original-graph-weight MST.
"""
from __future__ import annotations

from functools import lru_cache
import time

import numpy as np


METRICS = ("l1", "l2", "linf")
METHODS = {
    "l1": "MST-L1_EVA",
    "l2": "MST-L2_EVA",
    "linf": "MST-Linf_EVA",
}


@lru_cache(maxsize=1)
def _prim_kernel():
    try:
        from numba import njit
    except ImportError as exc:
        raise ImportError(
            "Exact complete-graph MST needs Numba. Install with "
            'python -m pip install "numba>=0.67,<0.69"'
        ) from exc

    @njit(cache=True)
    def prim(points, metric):
        n, p = points.shape
        selected = np.zeros(n, dtype=np.bool_)
        best = np.full(n, np.inf, dtype=np.float64)
        parent = np.full(n, -1, dtype=np.int32)
        best[0] = 0.0
        for _ in range(n):
            vertex = -1
            minimum = np.inf
            # First minimum in vertex-index order breaks ties reproducibly.
            for i in range(n):
                if not selected[i] and (vertex == -1 or best[i] < minimum):
                    vertex = i
                    minimum = best[i]
            selected[vertex] = True
            for j in range(n):
                if selected[j]:
                    continue
                if metric == 2:
                    # Squared Euclidean distance: same edge ordering as L2.
                    value = 0.0
                    for coordinate in range(p):
                        difference = points[vertex, coordinate] - points[j, coordinate]
                        value += difference * difference
                elif metric == 1:
                    value = 0.0
                    for coordinate in range(p):
                        value += abs(points[vertex, coordinate] - points[j, coordinate])
                else:
                    value = 0.0
                    for coordinate in range(p):
                        difference = abs(
                            points[vertex, coordinate] - points[j, coordinate]
                        )
                        if difference > value:
                            value = difference
                if value < best[j]:
                    best[j] = value
                    parent[j] = vertex
        return parent, best

    return prim


def _cut_longest_tree_edge(parent, edge_distances):
    """Produce the two connected components after removing one MST edge."""
    n = parent.size
    if n < 2 or parent[0] != -1 or np.any(parent[1:] < 0):
        raise ValueError("Invalid rooted MST parent array")
    # np.argmax chooses the lowest-index child on tied longest lengths.
    removed_child = int(np.argmax(edge_distances[1:]) + 1)
    children = [[] for _ in range(n)]
    for child in range(1, n):
        children[int(parent[child])].append(child)
    labels = np.zeros(n, dtype=np.int8)
    stack = [removed_child]
    while stack:
        vertex = stack.pop()
        labels[vertex] = 1
        stack.extend(children[vertex])
    counts = np.bincount(labels, minlength=2)
    if min(counts) < 1 or counts.sum() != n:
        raise RuntimeError("Removing one MST edge did not form two components")
    return labels, removed_child, tuple(int(v) for v in counts)


def mst_binary_partition(vectors, metric="l2"):
    """Return binary labels, diagnostics and the exact ambient-space MST.

    Distances are computed on all p input coordinates. This routine neither
    centers nor projects points and does not inspect graph weights.

    Returns
    -------
    labels : n-vector of 0/1
    report : dictionary with edge length, sizes, timings, complexity
    parent : MST parent indices (root 0 has parent -1)
    edge_distances : parent-to-child distances; root distance is 0

    One exact MST per metric/factor: repeated random seeds cannot change it.
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose {METRICS}")
    y = np.asarray(vectors, dtype=np.float64)
    if (y.ndim != 2 or y.shape[0] < 2 or y.shape[1] < 1
            or not np.isfinite(y).all()):
        raise ValueError("Expected finite n-by-p vectors with n >= 2 and p >= 1")
    y = np.ascontiguousarray(y)
    code = {"l1": 1, "l2": 2, "linf": 3}[metric]
    kernel = _prim_kernel()
    # Compile BEFORE timing the graph solve (warmup uses separate data).
    kernel(np.array([[0.0], [1.0]], dtype=np.float64), code)
    start = time.monotonic()
    parent, distances = kernel(y, code)
    if metric == "l2":
        # Compare squared lengths, then report ordinary Euclidean lengths.
        distances = np.sqrt(distances)
    labels, removed_child, counts = _cut_longest_tree_edge(parent, distances)
    elapsed = time.monotonic() - start
    if not np.isfinite(distances).all():
        raise ArithmeticError("MST has nonfinite edge distances")
    longest = float(distances[removed_child])
    return labels, {
        "method": METHODS[metric],
        "metric": metric,
        "ambient_dimensions": int(y.shape[1]),
        "vertices": int(y.shape[0]),
        "tree_edges": int(y.shape[0] - 1),
        "original_ambient_space": True,
        "pca_projection": False,
        "candidate_graph": "complete graph over EVA vectors",
        "distance_definition": {
            "l1": "sum(abs(x_i - y_i))",
            "l2": "sqrt(sum((x_i - y_i)^2))",
            "linf": "max(abs(x_i - y_i))",
        }[metric],
        "largest_edge_length": longest,
        "removed_edge_parent": int(parent[removed_child]),
        "removed_edge_child": removed_child,
        "component_sizes": list(counts),
        "mst_total_length": float(np.sum(distances)),
        "seconds_excluding_jit_compilation": float(elapsed),
        "is_randomized": False,
        "complexity": "O(n^2*p) time, O(n*p) memory",
    }, parent, distances
