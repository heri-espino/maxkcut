"""EVA: sequential greedy ascent of the Max-Cut vector relaxation.

Independent NumPy/SciPy reproduction of b3r8/EVA/EVA.m (Apache-2.0).
This is a NONCONVEX factorized optimization, not a dense convex SDP solve.
Retains all row vectors needed for GW and spherical k-means rounding.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import sparse


def _prepared_graph(adjacency):
    a = sparse.csr_matrix(adjacency, dtype=np.float64).copy()
    if a.shape[0] != a.shape[1] or a.shape[0] < 2:
        raise ValueError("EVA requires a square graph with at least 2 vertices")
    a.sum_duplicates()
    a.eliminate_zeros()
    if np.any(a.diagonal() != 0):
        raise ValueError("Self-loops are not allowed")
    delta = a - a.T
    if delta.nnz and np.max(np.abs(delta.data)) > 1e-10:
        raise ValueError("EVA requires symmetric edge weights")
    if not np.isfinite(a.data).all():
        raise ValueError("Nonfinite edge weight")
    return a


def vector_objective(a, vectors):
    """Exact continuous Max-Cut objective, counting each edge exactly once."""
    upper = sparse.triu(a, k=1, format="coo")
    dots = np.einsum(
        "ij,ij->i", vectors[upper.row], vectors[upper.col], optimize=True
    )
    return float(0.5 * np.dot(upper.data, 1.0 - dots))


def _identity(a, rank, seed, tolerance):
    digest = hashlib.sha256()
    for data in (a.indptr, a.indices, a.data):
        digest.update(np.ascontiguousarray(data).tobytes())
    return {
        "algorithm": "eva-sequential-python-v1",
        "graph_csr_sha256": digest.hexdigest(),
        "n": int(a.shape[0]),
        "p": int(rank),
        "seed": int(seed),
        "tolerance": float(tolerance),
        "dtype": "float64",
    }


def _write_checkpoint(path, vectors, state, rng):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".eva-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(
                stream, vectors=vectors,
                metadata=np.array(json.dumps({
                    **state, "rng_state": rng.bit_generator.state
                }, sort_keys=True)),
            )
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_eva_factor(path):
    """Read saved n-by-p vectors and validated metadata; no pickle execution."""
    with np.load(path, allow_pickle=False) as stored:
        vectors = stored["vectors"].copy()
        state = json.loads(str(stored["metadata"].item()))
    if (vectors.ndim != 2 or not np.isfinite(vectors).all()
            or np.any(np.abs(np.linalg.norm(vectors, axis=1) - 1) > 1e-6)):
        raise ValueError("EVA checkpoint contains invalid unit vectors")
    return vectors, state


def solve_eva(
    adjacency, *, rank=None, tolerance=1e-3, max_sweeps=10000,
    seed=2026, checkpoint_path=None, checkpoint_every=25,
    on_sweep=None,
):
    """Sequential greedy vector-coordinate ascent in float64.

    Each update is y_i := -sum_j(w_ij y_j) / ||sum_j(w_ij y_j)||.
    Uses the paper's positive-uniform initialization, sequential vertex
    order, angular stopping condition, and rank ceil(sqrt(2(n+1))).
    RNG draws differ from MATLAB; numerical trajectories need not coincide.

    Returns the n-by-p factor Y and diagnostics. On every periodic
    checkpoint, saves the factor *and RNG state*, enabling exact resume.
    """
    a = _prepared_graph(adjacency)
    n = a.shape[0]
    p = int(np.ceil(np.sqrt(2 * (n + 1)))) if rank is None else rank
    if not isinstance(p, (int, np.integer)) or not 1 <= p <= n:
        raise ValueError("rank must be an integer between 1 and n")
    if (not np.isfinite(tolerance) or tolerance <= 0
            or max_sweeps < 1 or checkpoint_every < 1 or seed < 0):
        raise ValueError("Invalid tolerance, sweep count, checkpoint interval or seed")
    identity = _identity(a, p, seed, tolerance)
    rng = np.random.default_rng(seed)
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    elapsed_prior = 0.0
    if checkpoint is not None and checkpoint.is_file():
        y, state = load_eva_factor(checkpoint)
        if state.get("identity") != identity or y.shape != (n, p):
            raise ValueError("EVA checkpoint graph/settings mismatch; use new output")
        rng.bit_generator.state = state["rng_state"]
        completed = int(state["sweeps"])
        elapsed_prior = float(state.get("elapsed_seconds", 0))
        if state.get("converged") or completed >= max_sweeps:
            return y, {k: v for k, v in state.items() if k != "rng_state"}
    else:
        y = rng.random((n, p), dtype=np.float64)
        y /= np.linalg.norm(y, axis=1)[:, None]
        completed = 0
        state = {
            "identity": identity, "sweeps": 0, "converged": False,
            "objective": vector_objective(a, y),
            "max_angle": None, "elapsed_seconds": 0.0,
        }
        if checkpoint is not None:
            _write_checkpoint(checkpoint, y, state, rng)

    started = time.monotonic()
    for sweep in range(completed + 1, max_sweeps + 1):
        max_angle = 0.0
        for i in range(n):
            low, high = a.indptr[i], a.indptr[i + 1]
            if high == low:
                continue
            field = a.data[low:high] @ y[a.indices[low:high]]
            norm = float(np.linalg.norm(field))
            if norm < 1e-15:
                next_y = rng.random(p)
                next_y /= np.linalg.norm(next_y)
            else:
                next_y = -field / norm
            cosine = float(np.clip(y[i] @ next_y, -1.0, 1.0))
            max_angle = max(max_angle, float(np.arccos(cosine)))
            y[i] = next_y
        objective = vector_objective(a, y)
        if objective < float(state["objective"]) - 1e-7 * max(1, abs(state["objective"])):
            raise ArithmeticError("EVA objective decreased beyond numerical tolerance")
        state = {
            "identity": identity, "sweeps": sweep,
            "converged": bool(max_angle < tolerance),
            "objective": objective, "max_angle": max_angle,
            "elapsed_seconds": elapsed_prior + time.monotonic() - started,
        }
        if on_sweep is not None:
            on_sweep(dict(state))
        if checkpoint is not None and (
            sweep % checkpoint_every == 0 or state["converged"]
            or sweep == max_sweeps
        ):
            _write_checkpoint(checkpoint, y, state, rng)
        if state["converged"]:
            break
    return y, state
