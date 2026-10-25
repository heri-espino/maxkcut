"""Sparse weighted Max-Cut one-vertex-flip local search.

A scalable *heuristic baseline*, not the Goemans-Williamson SDP algorithm.
It accepts negative edge weights and works directly on scipy CSR graphs.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse

from .io import edge_arrays
from .objective import cut_weight


def improve_by_flips(adjacency: sparse.spmatrix, labels: np.ndarray,
                     rng: np.random.Generator, *,
                     max_sweeps: int = 20, deadline: float | None = None):
    """Improve a binary partition monotonically using random-order flips.

    For spins s_i in {-1,+1}, flipping vertex i increases the weighted
    cut by g_i = s_i sum_j A_ij s_j. After a flip, gains at neighboring
    vertices are updated incrementally; no dense SDP matrix is formed.

    Only symmetric graphs with a zero diagonal are supported. Returns
    (partition, diagnostics). A deadline may interrupt a sweep; this
    returns a *feasible, improved* cut with status 'time_budget' rather
    than falsely claiming local optimality.
    """
    if not sparse.isspmatrix_csr(adjacency):
        adjacency = sparse.csr_matrix(adjacency)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != adjacency.shape[0] or not np.all(
        (labels == 0) | (labels == 1)
    ):
        raise ValueError("Expected a binary label for every graph vertex")
    if adjacency.shape[0] != adjacency.shape[1] or adjacency.diagonal().any():
        raise ValueError("Expected a square graph with zero diagonal")
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be positive")

    n = len(labels)
    spins = (1 - 2 * labels.astype(np.int8)).astype(np.float64)
    gains = spins * np.asarray(adjacency @ spins).ravel()
    edges = edge_arrays(adjacency)
    score_start = cut_weight(edges, labels)
    flips = 0
    sweeps = 0
    status = "sweep_limit"
    start = time.monotonic()

    for sweep in range(max_sweeps):
        accepted = 0
        # One complete sweep checks each vertex once; accepted moves always
        # have strictly positive gain at the moment they are applied.
        for t, index in enumerate(rng.permutation(n)):
            if t % 128 == 0 and deadline is not None and time.monotonic() >= deadline:
                status = "time_budget"
                break
            index = int(index)
            improvement = float(gains[index])
            if improvement <= 1e-10:
                continue
            old_spin = spins[index]
            first, last = adjacency.indptr[index:index + 2]
            neighbors = adjacency.indices[first:last]
            weights = adjacency.data[first:last]
            # Use the unchanged neighboring spins before changing s_i.
            gains[neighbors] -= 2.0 * old_spin * spins[neighbors] * weights
            gains[index] = -improvement
            spins[index] = -old_spin
            flips += 1
            accepted += 1

        sweeps += 1
        if status == "time_budget":
            break
        if accepted == 0:
            status = "local_optimum"
            break

    improved = (spins < 0).astype(np.int8)
    score_end = cut_weight(edges, improved)
    if score_end + 1e-8 < score_start:
        raise RuntimeError("Local search decreased the actual weighted cut")
    return improved, {
        "initial_cut": score_start,
        "cut_weight": score_end,
        "flips": flips,
        "sweeps": sweeps,
        "termination": status,
        "seconds": time.monotonic() - start,
    }


def random_flip_restart(adjacency: sparse.spmatrix, *,
                        seed: int, max_sweeps: int = 20,
                        deadline: float | None = None):
    """Generate one seeded partition and improve it by sparse flip search."""
    rng = np.random.default_rng(seed)
    start = rng.integers(0, 2, size=adjacency.shape[0], dtype=np.int8)
    return improve_by_flips(
        adjacency, start, rng, max_sweeps=max_sweeps, deadline=deadline
    )
