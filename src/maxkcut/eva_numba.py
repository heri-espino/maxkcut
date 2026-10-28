"""Optional compiled sequential EVA: same Gauss-Seidel update, less Python overhead.

This is intentionally NOT a GPU/Jacobi/block algorithm. Each row is
updated in place in the original vertex order. The optimization
criterion and angular convergence rule agree with EVA.m.

Experiment identity is segregated from the NumPy reference; never
overwrite/resume original EVA factors with a different backend.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .eva import (
    _identity, _prepared_graph, _write_checkpoint, load_eva_factor,
    vector_objective,
)


def _compiled_sweep():
    try:
        from numba import njit
    except ImportError as exc:
        raise ImportError(
            "Compiled EVA requires numba>=0.67 (NumPy 2.5 compatible). "
            "Install using: python -m pip install 'numba>=0.67,<0.69'. "
            "The NumPy reference backend is unchanged."
        ) from exc

    @njit(cache=True)
    def sweep(indptr, indices, weights, y, start, min_cos):
        """Return (objective gain, lowest angle cosine, first zero-field row).

        A negative-weight CSR edge is handled identically to a positive
        edge, including the signed local coordinate optimum.
        """
        n, p = y.shape
        h = np.empty(p, dtype=np.float64)
        gain = 0.0
        for i in range(start, n):
            first, last = indptr[i], indptr[i + 1]
            if first == last:
                continue
            for t in range(p):
                h[t] = 0.0
            for k in range(first, last):
                j = indices[k]
                w = weights[k]
                for t in range(p):
                    h[t] += w * y[j, t]
            magnitude_sq = 0.0
            dot = 0.0
            for t in range(p):
                magnitude_sq += h[t] * h[t]
                dot += y[i, t] * h[t]
            norm = np.sqrt(magnitude_sq)
            if norm < 1e-15:
                # Let Python's original, seeded NumPy RNG handle this rare
                # case without changing the reference RNG sequence.
                return gain, min_cos, i
            cosine = -dot / norm
            if cosine > 1.0:
                cosine = 1.0
            elif cosine < -1.0:
                cosine = -1.0
            if cosine < min_cos:
                min_cos = cosine
            gain += 0.5 * norm * (1.0 - cosine)
            for t in range(p):
                y[i, t] = -h[t] / norm
        return gain, min_cos, n

    return sweep


def solve_eva_numba(
    adjacency, *, rank=None, tolerance=1e-3, max_sweeps=10000,
    seed=2026, checkpoint_path=None, checkpoint_every=25,
    on_sweep=None,
):
    """Run compiled sequential EVA, preserving all factor/checkpoint semantics.

    Compile on a zero-row array *before* timing the numerical solve;
    report compilation separately, as MATLAB benchmark runtimes do not
    include Python JIT compilation. Warm-starts use this backend's
    dedicated experiment identity, never overwrite NumPy EVA files.
    """
    a = _prepared_graph(adjacency)
    a.sort_indices()
    n = a.shape[0]
    p = int(np.ceil(np.sqrt(2 * (n + 1)))) if rank is None else rank
    if not isinstance(p, (int, np.integer)) or not 1 <= p <= n:
        raise ValueError("rank must be an integer between 1 and n")
    if (not np.isfinite(tolerance) or tolerance <= 0
            or max_sweeps < 1 or checkpoint_every < 1 or seed < 0):
        raise ValueError("Invalid tolerance, sweep count, checkpoint interval or seed")
    identity = _identity(a, p, seed, tolerance)
    identity["algorithm"] = "eva-sequential-numba-v1"
    rng = np.random.default_rng(seed)
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    elapsed_prior = 0.0
    if checkpoint is not None and checkpoint.is_file():
        y, state = load_eva_factor(checkpoint)
        if state.get("identity") != identity or y.shape != (n, p):
            raise ValueError("EVA backend/graph/settings mismatch; use separate output")
        rng.bit_generator.state = state["rng_state"]
        completed = int(state["sweeps"])
        elapsed_prior = float(state.get("elapsed_seconds", 0.0))
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

    compile_start = time.monotonic()
    kernel = _compiled_sweep()
    # Trigger JIT compilation with matching array types and layout,
    # without advancing the actual optimization or changing its RNG.
    kernel(a.indptr[:1], a.indices[:0], a.data[:0],
           np.empty((0, p), dtype=np.float64), 0, 1.0)
    compile_seconds = time.monotonic() - compile_start

    started = time.monotonic()
    for sweep in range(completed + 1, max_sweeps + 1):
        lowest_cosine = 1.0
        delta = 0.0
        index = 0
        while index < n:
            added, lowest_cosine, zero_index = kernel(
                a.indptr, a.indices, a.data, y, index, lowest_cosine,
            )
            delta += added
            if zero_index == n:
                break
            # Identical fallback to the NumPy reference for zero-field rows.
            next_y = rng.random(p)
            next_y /= np.linalg.norm(next_y)
            cosine = float(np.clip(y[zero_index] @ next_y, -1.0, 1.0))
            if cosine < lowest_cosine:
                lowest_cosine = cosine
            y[zero_index] = next_y
            index = zero_index + 1

        objective = float(state["objective"]) + delta
        if objective < float(state["objective"]) - 1e-7 * max(
            1.0, abs(state["objective"])
        ):
            raise ArithmeticError("Compiled EVA objective decreased unexpectedly")
        max_angle = float(np.arccos(np.clip(lowest_cosine, -1.0, 1.0)))
        converged = max_angle < tolerance
        write_due = (sweep % checkpoint_every == 0 or converged
                     or sweep == max_sweeps)
        if write_due:
            # Independent exact objective audit catches arithmetic drift
            # and corrects accumulated floating point error in checkpoints.
            exact = vector_objective(a, y)
            if not np.isclose(exact, objective, atol=1e-5, rtol=1e-8):
                raise ArithmeticError(
                    f"Compiled EVA incremental objective disagrees with exact: "
                    f"{objective} vs {exact}"
                )
            objective = exact
        state = {
            "identity": identity, "sweeps": sweep,
            "converged": bool(converged),
            "objective": objective, "max_angle": max_angle,
            "elapsed_seconds": elapsed_prior + time.monotonic() - started,
            "compile_seconds": compile_seconds,
        }
        if checkpoint is not None and write_due:
            _write_checkpoint(checkpoint, y, state, rng)
        if on_sweep is not None:
            on_sweep(dict(state))
        if converged:
            break
    return y, state
