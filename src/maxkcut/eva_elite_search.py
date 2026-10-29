"""Experiment011: large-budget elite diversification over saved KL Max-Cut cuts.

This module never runs EVA, rounding, or an MST. It consumes already
verified experiment010 KL output partitions, always scores ORIGINAL signed
undirected graph edges, and only returns the best feasible cut observed.

Algorithms:
  ILS-KL       : repeated random + near-boundary perturbation, fast KL,
                 simulated-annealing-style exploration, incumbent retention.
  EliteMix-KL  : complement-aligned elite solution recombination + fast KL.
  Tabu-KL      : best-admissible Tabu bursts followed by fast KL.

The fast KL pass uses a Numba-compiled indexed max-heap over vertex flip
gains. Unlike the earlier O(n^2) argmax scan per KL pass, it updates only
flipped vertices' neighbors and costs O((n+m)log n) worst-case per pass.
It implements the SAME greedy locked-gain / best cumulative prefix rule
as src/maxkcut/eva_escape.py, with fixed RNG order for ties.
"""
from __future__ import annotations

from functools import lru_cache
import time

import numpy as np
from scipy import sparse

from .eva_escape import _preflight, optimize_tabu
from .io import edge_arrays
from .objective import cut_weight

METHODS = ("ILS-KL", "EliteMix-KL", "Tabu-KL")


@lru_cache(maxsize=1)
def _fast_kl_kernel():
    try:
        from numba import njit
    except ImportError as exc:
        raise ImportError(
            'Experiment011 requires Numba: pip install "numba>=0.67,<0.69"'
        ) from exc

    @njit(cache=True)
    def better(left, right, gain, rank):
        # Deterministic randomized tie handling identical to the
        # original KL full-scan algorithm using seeded permutation.
        if gain[left] > gain[right]:
            return True
        if gain[left] < gain[right]:
            return False
        return rank[left] < rank[right]

    @njit(cache=True)
    def sift_down(heap, positions, gain, rank, i, size):
        while True:
            left = 2 * i + 1
            if left >= size:
                return
            right = left + 1
            best = left
            if right < size and better(heap[right], heap[left], gain, rank):
                best = right
            if not better(heap[best], heap[i], gain, rank):
                return
            a, b = heap[i], heap[best]
            heap[i], heap[best] = b, a
            positions[a] = best
            positions[b] = i
            i = best

    @njit(cache=True)
    def sift_up(heap, positions, gain, rank, i):
        while i > 0:
            p = (i - 1) // 2
            if not better(heap[i], heap[p], gain, rank):
                return
            a, b = heap[i], heap[p]
            heap[i], heap[p] = b, a
            positions[a] = p
            positions[b] = i
            i = p

    @njit(cache=True)
    def kernel(indptr, indices, weights, spins, order):
        n = len(spins)
        gain = np.zeros(n, dtype=np.float64)
        for i in range(n):
            subtotal = 0.0
            for ptr in range(indptr[i], indptr[i + 1]):
                subtotal += weights[ptr] * spins[indices[ptr]]
            gain[i] = spins[i] * subtotal
        rank = np.empty(n, dtype=np.int32)
        for pos in range(n):
            rank[order[pos]] = pos
        heap = np.arange(n, dtype=np.int32)
        positions = np.arange(n, dtype=np.int32)
        for pos in range(n // 2 - 1, -1, -1):
            sift_down(heap, positions, gain, rank, pos, n)

        selected = np.empty(n, dtype=np.int32)
        sum_gain = 0.0
        best_gain = 0.0
        best_prefix = 0
        for step in range(n):
            size = n - step
            node = heap[0]
            selected[step] = node
            node_gain = gain[node]
            positions[node] = -1
            if size > 1:
                replace = heap[size - 1]
                heap[0] = replace
                positions[replace] = 0
                sift_down(heap, positions, gain, rank, 0, size - 1)
            old_spin = spins[node]
            for ptr in range(indptr[node], indptr[node + 1]):
                neighbor = indices[ptr]
                neighbor_pos = positions[neighbor]
                if neighbor_pos < 0:
                    continue  # neighbor was locked and removed already
                previous = gain[neighbor]
                gain[neighbor] = previous - (
                    2.0 * weights[ptr] * old_spin * spins[neighbor]
                )
                if gain[neighbor] > previous:
                    sift_up(heap, positions, gain, rank, neighbor_pos)
                else:
                    sift_down(heap, positions, gain, rank, neighbor_pos, size - 1)
            spins[node] = -old_spin
            sum_gain += node_gain
            if sum_gain > best_gain + 1e-10:
                best_gain = sum_gain
                best_prefix = step + 1
        # Keep only the best improving prefix; restore all later moves.
        for step in range(n - 1, best_prefix - 1, -1):
            node = selected[step]
            spins[node] = -spins[node]
        return best_gain, best_prefix

    return kernel


def optimize_kl_fast(adjacency, labels, rng, *, passes=15):
    """Fast exact greedy KL gain-selection rule (not global Max-Cut)."""
    if passes < 1:
        raise ValueError("passes must be >= 1")
    adjacency, spins, edges = _preflight(adjacency, labels)
    score_before = float(cut_weight(edges, labels))
    kernel = _fast_kl_kernel()
    started = time.monotonic()
    gains = []
    flips = 0
    termination = "pass_limit"
    for _ in range(passes):
        order = np.asarray(rng.permutation(len(spins)), dtype=np.int32)
        gain, prefix = kernel(
            adjacency.indptr, adjacency.indices, adjacency.data, spins, order,
        )
        if gain <= 1e-10:
            termination = "no_improving_gain_sequence"
            break
        gains.append(float(gain))
        flips += int(prefix)
    final = (spins < 0).astype(np.int8)
    score_after = float(cut_weight(edges, final))
    if (score_after + 1e-7 < score_before
            or not np.isclose(score_after - score_before, sum(gains),
                              rtol=0, atol=1e-6)):
        raise RuntimeError("Indexed-heap KL gain accounting disagrees with Max-Cut")
    return final, {
        "initial_cut": score_before, "cut_weight": score_after,
        "improvement": score_after-score_before,
        "passes_improved": len(gains),
        "flips_retained": flips,
        "termination": termination,
        "seconds": time.monotonic() - started,
    }


def _kick(adjacency, labels, rng, count, boundary_fraction):
    """Diversify by flipping a random set, often near zero flip gain."""
    n = len(labels)
    count = max(1, min(int(count), n))
    boundary_count = min(count, int(round(count * boundary_fraction)))
    if boundary_count:
        spins = 1.0 - 2.0 * labels.astype(float)
        gains = spins * np.asarray(adjacency @ spins).ravel()
        # Include many moderately ambiguous vertices; avoid fully
        # deterministic repeated selection of the same boundary set.
        width = min(n, max(count, 8 * count))
        pool = np.argpartition(np.abs(gains), width - 1)[:width]
        chosen = rng.choice(pool, size=boundary_count, replace=False)
    else:
        chosen = np.empty(0, dtype=np.int32)
    free = count - len(chosen)
    if free:
        selectable = np.ones(n, dtype=bool)
        selectable[chosen] = False
        rest = rng.choice(np.flatnonzero(selectable), size=free, replace=False)
        chosen = np.concatenate((chosen, rest))
    new_labels = np.asarray(labels, dtype=np.int8).copy()
    new_labels[chosen] ^= 1
    return new_labels


def _aligned_partner(reference, partner):
    """Binary cut invariant under global complement; choose closer form."""
    reference = np.asarray(reference, dtype=np.int8)
    partner = np.asarray(partner, dtype=np.int8)
    if np.count_nonzero(reference != partner) > reference.size // 2:
        return (1 - partner).astype(np.int8)
    return partner.copy()


def _elite_mix(reference, partner, rng, fraction):
    """Combine two existing cut labelings, with polarity aligned."""
    partner = _aligned_partner(reference, partner)
    candidates = np.flatnonzero(reference != partner)
    if len(candidates) == 0:
        return reference.copy(), 0
    n = max(1, min(len(candidates), int(round(fraction * len(candidates)))))
    changed = rng.choice(candidates, size=n, replace=False)
    proposal = reference.copy()
    proposal[changed] = partner[changed]
    return proposal, int(len(changed))


def _validate_elites(adjacency, elites):
    if len(elites) < 1:
        raise ValueError("At least one elite warm start is required")
    n = adjacency.shape[0]
    for labels, score in elites:
        if (np.asarray(labels).shape != (n,)
                or not np.all((labels == 0) | (labels == 1))):
            raise ValueError("Invalid elite binary partition shape")
        actual = float(cut_weight(edge_arrays(adjacency), labels))
        if not np.isclose(actual, score, atol=1e-8, rtol=0):
            raise ValueError("Elite starting score does not match original signed graph")


def optimize_elite(
    adjacency, initial, elites, rng, *,
    method="ILS-KL", cycles=80, kl_passes=12,
    kick_sizes=(8, 16, 32, 64, 128, 256),
    boundary_fraction=0.65, temperature=12.0,
    tabu_steps=10000, tabu_tenure=15,
    tabu_stagnation=3000, target=None,
):
    """Run a complete independently seeded diversification episode.

    All strategies retain the highest verified score encountered, and
    continue after local stagnation until cycles have elapsed, unless
    an explicitly requested target has been reached.

    ILS-KL: variable perturbation of incumbent or walking local state.
    EliteMix-KL: fresh incumbent + randomized complement-aligned mix of
      a second source elite, followed by KL.
    Tabu-KL: short randomized tenure Tabu from incumbent/walk plus KL.

    The starting partition and all elites are independently verified.
    Each episode returns a trace of *every* outer cycle plus initial
    and final score, allowing analysis of budget and convergence.
    """
    if method not in METHODS:
        raise ValueError("Unknown elite search strategy")
    if (cycles < 1 or kl_passes < 1 or len(kick_sizes) < 1
            or any(not 1 <= int(k) <= adjacency.shape[0] for k in kick_sizes)
            or not 0 <= boundary_fraction <= 1
            or temperature < 0 or tabu_steps < 1
            or tabu_tenure < 1 or tabu_stagnation < 1):
        raise ValueError("Invalid elite-search budget or perturbation settings")

    graph = sparse.csr_matrix(adjacency).astype(np.float64)
    graph.sort_indices()
    edges = edge_arrays(graph)
    original = np.asarray(initial, dtype=np.int8).copy()
    _validate_elites(graph, elites)
    initial_score = float(cut_weight(edges, original))
    best_labels = original.copy()
    best_score = initial_score
    walk_labels = original.copy()
    walk_score = initial_score
    started = time.monotonic()
    trace = []
    best_events = []
    method_label = method

    for cycle in range(cycles):
        if method_label == "ILS-KL":
            base = best_labels if rng.random() < 0.72 else walk_labels
            kick = int(kick_sizes[cycle % len(kick_sizes)])
            # Mild reproducible jitter prevents repeatedly testing the
            # exact same Hamming sphere around the current incumbent.
            kick = min(len(base), max(1, int(round(
                kick * rng.uniform(0.8, 1.2)
            ))))
            proposal = _kick(
                graph, base, rng, kick, boundary_fraction,
            )
            transition = f"perturb-{kick}"
        elif method_label == "EliteMix-KL":
            # Deterministically seeded random elite partner and mixing
            # proportions. Complement alignment is crucial: equivalent
            # cuts may otherwise appear maximally different.
            p = int(rng.integers(0, len(elites)))
            reference = best_labels if rng.random() < 0.75 else walk_labels
            fraction = float(rng.choice(
                np.array([0.15, 0.3, 0.5, 0.7, 0.85]),
            ))
            proposal, hamming = _elite_mix(
                reference, np.asarray(elites[p][0], dtype=np.int8),
                rng, fraction,
            )
            if hamming == 0:
                proposal = _kick(
                    graph, reference, rng,
                    int(kick_sizes[cycle % len(kick_sizes)]),
                    boundary_fraction,
                )
            transition = f"elite-{p}-mix-{hamming}"
        else:
            reference = best_labels if rng.random() < 0.75 else walk_labels
            # Rotate Tabu tenure to escape the persistent 26,688 plateau
            # observed at a single tenure=25 on G63.
            tenure = max(2, int(round(tabu_tenure * (
                (0.5, 1.0, 1.5, 2.0)[cycle % 4]
            ))))
            proposal, tabu_diag = optimize_tabu(
                graph, reference, rng, steps=tabu_steps,
                tenure=tenure, stagnation=tabu_stagnation,
            )
            transition = f"tabu-tenure-{tenure}-steps-{tabu_diag['steps']}"
        next_labels, kl = optimize_kl_fast(
            graph, proposal, rng, passes=kl_passes,
        )
        score = float(cut_weight(edges, next_labels))
        if score + 1e-8 < float(cut_weight(edges, proposal)):
            raise AssertionError("KL must not decrease proposal Max-Cut")
        if score > best_score + 1e-8:
            best_labels = next_labels.copy()
            best_score = score
            best_events.append({
                "cycle": cycle + 1,
                "new_best_cut": score,
            })
        # Keep a wandering state that can cross low valleys, but never
        # lose the best-so-far archived partition.
        if score >= walk_score - 1e-10:
            accepted = True
        elif temperature > 0:
            temp = temperature * max(0.1, 1.0 - cycle / cycles)
            accepted = bool(rng.random() < np.exp(
                np.clip((score-walk_score) / temp, -700, 0)
            ))
        else:
            accepted = False
        if accepted:
            walk_labels = next_labels.copy()
            walk_score = score
        trace.append({
            "cycle": cycle + 1,
            "transition": transition,
            "candidate_cut": score,
            "best_cut": best_score,
            "accepted_walk": accepted,
            "kl_passes_improved": kl["passes_improved"],
            "kl_seconds": kl["seconds"],
        })
        if target is not None and best_score >= target:
            break
    final_score = float(cut_weight(edges, best_labels))
    if (final_score + 1e-8 < initial_score
            or final_score != best_score
            or not all(
                x["best_cut"] + 1e-8 >= initial_score for x in trace
            )):
        raise AssertionError("Elite refinement lost its best signed Max-Cut")
    return best_labels, {
        "method": method,
        "initial_cut": initial_score,
        "cut_weight": final_score,
        "improvement": final_score - initial_score,
        "cycles_completed": len(trace),
        "best_events": best_events,
        "trace": trace,
        "termination": (
            "target_reached" if target is not None and final_score >= target
            else "cycle_limit"
        ),
        "seconds": time.monotonic() - started,
    }
