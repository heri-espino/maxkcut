"""Numba-accelerated escape heuristics for signed weighted binary Max-Cut.

Both solvers use the original undirected sparse CSR adjacency (NOT the
EVA embedding). They optimize a previously saved 0/1 partition and
return the *best visited* partition, never a lower-scoring last iterate.

"Kernighan-Lin-style": a generalized single-vertex gain-sequence
pass, rather than classic balanced-pair swaps. Each vertex is locked
after being selected once in the pass; the highest gain unlocked move
may be NEGATIVE. Keep only the best strictly positive cumulative prefix
of that pass. Multiple passes may escape a 1-flip local optimum via
negative-gain moves with a net positive sequence.

"Tabu": best-admissible signed gain move at each iteration; temporary
vertex-move prohibition and best-so-far aspiration allow unfavorable
moves. A nonimprovement stopping limit bounds computation.

For spins s_i in {-1, +1}, flipping vertex i yields
  gain_i = s_i * sum_j w_ij * s_j,
and after flipping i:
  gain_j' = gain_j - 2*w_ij*s_i*s_j for each neighbor j.

All objective checks and final ranking use the original signed graph.
The Numba dependency is needed ONLY when these optional methods run.
"""
from __future__ import annotations

from functools import lru_cache
import time

import numpy as np
from scipy import sparse

from .io import edge_arrays
from .objective import cut_weight


METHODS = ("KL", "Tabu")


@lru_cache(maxsize=1)
def _kernels():
    try:
        from numba import njit
    except ImportError as exc:
        raise ImportError(
            'KL/Tabu acceleration requires Numba; install '
            'python -m pip install "numba>=0.67,<0.69"'
        ) from exc

    @njit(cache=True)
    def _initial_gains(indptr, indices, weights, spins):
        n = spins.size
        gains = np.zeros(n, dtype=np.float64)
        for i in range(n):
            summation = 0.0
            for ptr in range(indptr[i], indptr[i + 1]):
                summation += weights[ptr] * spins[indices[ptr]]
            gains[i] = spins[i] * summation
        return gains

    @njit(cache=True)
    def _kl_pass(indptr, indices, weights, spins, order):
        """Perform a full gain sequence, restore suffix beyond best prefix."""
        n = spins.size
        gains = _initial_gains(indptr, indices, weights, spins)
        locked = np.zeros(n, dtype=np.uint8)
        chosen = np.empty(n, dtype=np.int32)
        cumulative = 0.0
        best_cumulative = 0.0
        best_prefix = 0
        for pos in range(n):
            move = -1
            gain = -np.inf
            for order_index in range(n):
                candidate = order[order_index]
                value = gains[candidate]
                if locked[candidate] == 0 and value > gain:
                    move = candidate
                    gain = value
            if move < 0:
                raise RuntimeError("No available unlocked vertex")
            chosen[pos] = move
            locked[move] = 1
            old_spin = spins[move]
            for ptr in range(indptr[move], indptr[move + 1]):
                neighbor = indices[ptr]
                gains[neighbor] -= (
                    2.0 * weights[ptr] * old_spin * spins[neighbor]
                )
            spins[move] = -old_spin
            gains[move] = -gain
            cumulative += gain
            if cumulative > best_cumulative + 1e-10:
                best_cumulative = cumulative
                best_prefix = pos + 1
        # Restore the non-improving suffix in reverse; only retained
        # prefix matters since next pass rebuilds all sparse gains.
        for pos in range(n - 1, best_prefix - 1, -1):
            move = chosen[pos]
            spins[move] = -spins[move]
        return best_cumulative, best_prefix

    @njit(cache=True)
    def _tabu_search(indptr, indices, weights, spins, order, jitters,
                     steps_limit, tenure, no_improve_limit):
        n = spins.size
        gains = _initial_gains(indptr, indices, weights, spins)
        tabu_until = np.zeros(n, dtype=np.int64)
        best_spins = spins.copy()
        current_delta = 0.0
        best_delta = 0.0
        last_improvement = 0
        improvements = 0
        steps_done = 0
        negative_moves = 0
        aspiration_moves = 0
        for step in range(steps_limit):
            chosen = -1
            chosen_gain = -np.inf
            chosen_aspiration = False
            for idx in range(n):
                vertex = order[idx]
                gain = gains[vertex]
                aspired = current_delta + gain > best_delta + 1e-10
                if step >= tabu_until[vertex] or aspired:
                    if gain > chosen_gain:
                        chosen = vertex
                        chosen_gain = gain
                        chosen_aspiration = (
                            step < tabu_until[vertex] and aspired
                        )
            if chosen < 0:
                break
            if chosen_gain <= 1e-10:
                negative_moves += 1
            if chosen_aspiration:
                aspiration_moves += 1
            old_spin = spins[chosen]
            for ptr in range(indptr[chosen], indptr[chosen + 1]):
                neighbor = indices[ptr]
                gains[neighbor] -= (
                    2.0 * weights[ptr] * old_spin * spins[neighbor]
                )
            spins[chosen] = -old_spin
            gains[chosen] = -chosen_gain
            current_delta += chosen_gain
            tabu_until[chosen] = step + tenure + jitters[step] + 1
            steps_done += 1
            if current_delta > best_delta + 1e-10:
                best_delta = current_delta
                for i in range(n):
                    best_spins[i] = spins[i]
                improvements += 1
                last_improvement = steps_done
            if steps_done - last_improvement >= no_improve_limit:
                break
        return (best_spins, best_delta, steps_done, negative_moves,
                improvements, aspiration_moves,
                steps_done - last_improvement >= no_improve_limit)

    return _kl_pass, _tabu_search


def _preflight(adjacency, labels):
    if not sparse.isspmatrix_csr(adjacency):
        adjacency = sparse.csr_matrix(adjacency)
    if (adjacency.shape[0] != adjacency.shape[1]
            or np.any(adjacency.diagonal() != 0)):
        raise ValueError("Expected square, zero-diagonal graph")
    if (adjacency - adjacency.T).nnz:
        raise ValueError("Expected symmetric signed weighted adjacency")
    adjacency = adjacency.astype(np.float64)
    adjacency.sort_indices()
    labels = np.asarray(labels, dtype=np.int8)
    if (labels.ndim != 1 or len(labels) != adjacency.shape[0]
            or not np.all((labels == 0) | (labels == 1))):
        raise ValueError("Expected exactly one binary label per graph vertex")
    spins = (1.0 - 2.0 * labels.astype(np.float64))
    return adjacency, spins, edge_arrays(adjacency)


def optimize_kl(adjacency, labels, rng, *, passes=20):
    """Unbalanced KL-style best-prefix gain sequence on a sparse signed graph.

    O(passes*(n^2+m)) time with Numba; O(n+m) memory.
    All chosen prefix gains are strictly positive in aggregate. The
    labels may be locally optimal in the 1-flip neighborhood yet still
    gain from a sequence including temporarily negative flips.
    """
    if passes < 1:
        raise ValueError("passes must be >= 1")
    adjacency, spins, edges = _preflight(adjacency, labels)
    start_score = float(cut_weight(edges, labels))
    kernel, _ = _kernels()
    started = time.monotonic()
    gains_by_pass = []
    flips = 0
    reason = "pass_limit"
    for _ in range(passes):
        order = np.asarray(rng.permutation(len(spins)), dtype=np.int32)
        gain, count = kernel(
            adjacency.indptr, adjacency.indices, adjacency.data,
            spins, order,
        )
        if gain <= 1e-10:
            reason = "no_improving_gain_sequence"
            break
        gains_by_pass.append(float(gain))
        flips += int(count)
    improved = (spins < 0).astype(np.int8)
    final_score = float(cut_weight(edges, improved))
    if final_score + 1e-8 < start_score:
        raise RuntimeError("KL reduced the actual signed weighted Max-Cut")
    if not np.isclose(
        final_score-start_score, sum(gains_by_pass), rtol=0, atol=1e-6,
    ):
        raise RuntimeError("KL gain sequence and independently rescored cut disagree")
    return improved, {
        "initial_cut": start_score,
        "cut_weight": final_score,
        "improvement": final_score-start_score,
        "passes_improved": len(gains_by_pass),
        "flips_retained": flips,
        "gain_by_pass": gains_by_pass,
        "termination": reason,
        "seconds": time.monotonic()-started,
    }


def optimize_tabu(adjacency, labels, rng, *, steps=40000,
                  tenure=25, stagnation=8000):
    """Best-admissible vertex Tabu, aspiration and incumbent tracking.

    O(steps*(n+average_degree)) time with Numba; O(n+m) memory.
    Returns the best observed feasible cut (not the possibly worse
    last state). Random tie order and tenure jitter are seeded.
    """
    if steps < 1 or tenure < 1 or stagnation < 1:
        raise ValueError("steps, tenure and stagnation must be >= 1")
    adjacency, spins, edges = _preflight(adjacency, labels)
    start_score = float(cut_weight(edges, labels))
    _, kernel = _kernels()
    order = np.asarray(rng.permutation(len(spins)), dtype=np.int32)
    jitter_range = max(1, tenure // 4)
    jitters = rng.integers(0, jitter_range+1, size=steps, dtype=np.int32)
    started = time.monotonic()
    best, gain, done, negative, improvements, aspiration, stopped = kernel(
        adjacency.indptr, adjacency.indices, adjacency.data, spins,
        order, jitters, steps, tenure, stagnation,
    )
    improved = (best < 0).astype(np.int8)
    final_score = float(cut_weight(edges, improved))
    if final_score + 1e-8 < start_score:
        raise RuntimeError("Tabu returned a worse cut than its starting partition")
    if not np.isclose(final_score-start_score, gain, rtol=0, atol=1e-6):
        raise RuntimeError("Tabu incremental gains disagree with actual Max-Cut")
    return improved, {
        "initial_cut": start_score,
        "cut_weight": final_score,
        "improvement": final_score-start_score,
        "steps": int(done),
        "nonpositive_moves": int(negative),
        "new_best_events": int(improvements),
        "aspiration_moves": int(aspiration),
        "tenure": int(tenure),
        "stagnation": int(stagnation),
        "termination": "stagnation" if stopped else "step_limit",
        "seconds": time.monotonic()-started,
    }
