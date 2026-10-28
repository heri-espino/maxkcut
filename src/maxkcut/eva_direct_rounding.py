"""Direct optimization of GW ROUNDING directions, with EVA factor frozen.

For binary labels produced by a hyperplane normal r in R^p,

    x_i(r) = 1[Y_i @ r > 0]
    C(r) = sum_{i<j} w_ij * 1[x_i(r) != x_j(r)].

We optimize r itself, never Y and never K-means. For any unit tangent
direction v perpendicular to r, the great semicircle

    r(theta) = cos(theta)*r + sin(theta)*v, theta in (-pi/2,pi/2)

induces only finitely many changes to labels. Vertex i changes sides
at theta_i = atan(-(Y_i @ r)/(Y_i @ v)), provided Y_i@v != 0.
Sort these breakpoints; process all equal-angle events together.
The exact best Max-Cut labeling along this fixed semicircle follows
from signed flip-gain updates in O(m + n log n) plus two projections.

Repeating EXACT line sweeps along seeded random and objective-guided
tangent directions is monotone ascent over the piecewise constant
original weighted signed graph cut, NOT a surrogate differentiable
objective, NOT global optimization, and NOT unrestricted Max-Cut.
"""
from __future__ import annotations

from functools import lru_cache
import time

import numpy as np
from scipy import sparse

from .io import edge_arrays
from .objective import cut_weight


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    return None if norm <= 1e-13 or not np.isfinite(norm) else v / norm


@lru_cache(maxsize=1)
def _sweep_kernel():
    try:
        from numba import njit
    except ImportError as exc:
        raise ImportError("Experiment015 direct rounding requires numba") from exc

    @njit(cache=True)
    def sweep(indptr, indices, weights, first_labels, ordered_vertices,
              event_angles, initial_cut):
        n = first_labels.size
        spins = 1.0 - 2.0 * first_labels.astype(np.float64)
        gains = np.empty(n, dtype=np.float64)
        for i in range(n):
            field = 0.0
            for k in range(indptr[i], indptr[i+1]):
                field += weights[k] * spins[indices[k]]
            gains[i] = spins[i] * field

        best_score = initial_cut
        best_prefix = 0
        score = initial_cut
        count = ordered_vertices.size
        prefix = 0
        while prefix < count:
            # Equal-root vertices must flip as a WHOLE group.
            angle = event_angles[prefix]
            while prefix < count and event_angles[prefix] == angle:
                i = ordered_vertices[prefix]
                gain = gains[i]
                score += gain
                old_spin = spins[i]
                for k in range(indptr[i], indptr[i+1]):
                    j = indices[k]
                    gains[j] -= 2.0 * weights[k] * old_spin * spins[j]
                spins[i] = -old_spin
                gains[i] = -gain
                prefix += 1
            if score > best_score + 1e-8:
                best_score = score
                best_prefix = prefix
        return best_prefix, best_score

    return sweep


def optimize_semicircle(graph, vectors, normal, tangent):
    """Exact signed-Max-Cut line sweep for a specified great semicircle.

    Floating-point near-degenerate angles may not permit reliable
    reconstruction of a predicted interval. In that exceptional case
    return the original direction without claiming a better realized cut.
    """
    y = np.asarray(vectors, dtype=np.float64)
    r = _unit(normal)
    if r is None or y.ndim != 2 or y.shape[1] != r.size:
        raise ValueError("Invalid EVA matrix or rounding normal")
    if graph.shape != (y.shape[0], y.shape[0]):
        raise ValueError("Graph and EVA factor vertex counts disagree")
    v = np.asarray(tangent, dtype=np.float64)
    v = _unit(v - float(v@r)*r)
    if v is None:
        raise ValueError("Tangent direction must be independent of normal")
    a, b = y@r, y@v
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite projected EVA factor")
    graph_edges = edge_arrays(graph)
    baseline_labels = (a > 0).astype(np.int8)
    baseline = float(cut_weight(graph_edges, baseline_labels))

    # At theta -> -pi/2, the sign of -b determines the first cut.
    # Rows with exactly b=0 keep the sign of a throughout the open arc.
    first_labels = np.zeros(len(a), dtype=np.int8)
    first_labels[b < 0] = 1
    fixed = b == 0.
    first_labels[fixed] = (a[fixed] > 0).astype(np.int8)
    active = np.flatnonzero(~fixed)
    angles = np.arctan(-a[active]/b[active])
    ordering = np.argsort(angles, kind="stable")
    ordered_vertices = active[ordering].astype(np.int32)
    event_angles = np.asarray(angles[ordering], dtype=np.float64)
    initial_cut = float(cut_weight(graph_edges, first_labels))
    csr = sparse.csr_matrix(graph)
    csr.sort_indices()
    prefix, predicted = _sweep_kernel()(
        csr.indptr, csr.indices, csr.data,
        first_labels, ordered_vertices, event_angles, initial_cut,
    )
    expected_partition = first_labels.copy()
    expected_partition[ordered_vertices[:prefix]] ^= 1
    audited = float(cut_weight(graph_edges, expected_partition))
    if not np.isclose(audited, predicted, atol=1e-6, rtol=0):
        raise RuntimeError("Exact signed sweep gain increments are inconsistent")

    # Pick an interior angle in the best event-free interval.
    lo = float(event_angles[prefix-1]) if prefix else -np.pi/2
    hi = float(event_angles[prefix]) if prefix < len(event_angles) else np.pi/2
    theta = float(0.5 * lo + 0.5 * hi)
    candidate = _unit(np.cos(theta)*r + np.sin(theta)*v)
    if candidate is None:
        raise RuntimeError("Line sweep produced invalid rounded direction")
    actual_labels = (y@candidate > 0).astype(np.int8)
    final_cut = float(cut_weight(graph_edges, actual_labels))
    numerically_valid = bool(
        np.array_equal(actual_labels, expected_partition)
        and np.isclose(final_cut, predicted, atol=1e-6, rtol=0)
    )
    if not numerically_valid:
        # E.g. projections near machine-zero causing a direction/cut
        # mismatch. Never claim the unattainable sweep score.
        candidate = r.copy()
        actual_labels = baseline_labels.copy()
        final_cut = baseline
    if final_cut + 1e-8 < baseline:
        # The current r is a legal angle=0 point on the SAME arc.
        # Using it as a fallback makes exact realized cut monotone.
        candidate = r.copy()
        actual_labels = baseline_labels.copy()
        final_cut = baseline
    return candidate, actual_labels, {
        "initial_cut": baseline,
        "predicted_arc_max": float(predicted),
        "realized_arc_cut": final_cut,
        "improvement": final_cut - baseline,
        "events": len(active),
        "distinct_events": int(np.unique(event_angles).size),
        "numerically_valid": numerically_valid,
        "theta": theta,
        "selected_prefix": int(prefix),
    }


def _guided_tangent(graph, y, r, rng, mode, boundary_vertices=96):
    if mode == "random":
        vector = rng.standard_normal(r.size)
        return _unit(vector - (vector@r)*r)
    spins = np.where(y@r > 0., -1., 1.)
    gains = spins * np.asarray(graph@spins).ravel()
    if mode == "gain":
        bad = np.flatnonzero(gains < -1e-9)
        if not len(bad):
            mode = "boundary"
        else:
            # Rotate hyperplane so genuinely improving single-label
            # flips become more likely, but do not force any flips;
            # the exact signed cut line sweep determines acceptance.
            coeff = -spins[bad] * (-gains[bad])
            vector = y[bad].T@coeff
            return _unit(vector - (vector@r)*r)
    if mode == "boundary":
        size = min(y.shape[0], boundary_vertices)
        candidate_vertices = np.argpartition(
            np.abs(gains), size-1
        )[:size]
        signs = rng.choice(np.array([-1., 1.]), size=size)
        vector = y[candidate_vertices].T@signs
        return _unit(vector - (vector@r)*r)
    raise ValueError("Unknown tangent generator")


def optimize_hyperplane(
    graph, vectors, rng, *, initial=None, directions=250,
    modes=("random", "gain", "boundary"), target=None,
):
    """Iterated exact-line-search ascent on the original rounded Max-Cut.

    Return best binary hyperplane cut and its unit normal, with
    complete per-attempt score trace for independent replication.
    The factor is NEVER modified. Tie-breaking and RNG stream are
    independent of worker arrival order.
    """
    y = np.asarray(vectors, dtype=np.float64)
    if (y.ndim != 2 or y.shape[0] != graph.shape[0] or y.shape[1] < 1
            or not np.isfinite(y).all()):
        raise ValueError("Invalid original EVA factor")
    if (directions < 1 or not modes or
            any(m not in ("random", "gain", "boundary") for m in modes)):
        raise ValueError("Invalid direct-rounding search budget")
    r = _unit(rng.standard_normal(y.shape[1]) if initial is None else initial)
    if r is None:
        raise ValueError("Invalid initial hyperplane direction")
    edges = edge_arrays(graph)
    start = time.monotonic()
    first_labels = (y@r > 0).astype(np.int8)
    first_score = float(cut_weight(edges, first_labels))
    best_labels, best_r = first_labels.copy(), r.copy()
    best_score = first_score
    accepted = 0
    trace = []
    for iteration in range(directions):
        mode = modes[iteration % len(modes)]
        if mode == "random":
            proposal = rng.standard_normal(r.size)
            proposal -= (proposal@r)*r
            v = _unit(proposal)
        else:
            v = _guided_tangent(graph, y, r, rng, mode)
        if v is None:
            proposal = rng.standard_normal(r.size)
            proposal -= (proposal@r)*r
            v = _unit(proposal)
        if v is None:
            raise ValueError("Unable to propose any line-search tangent")
        next_r, next_labels, info = optimize_semicircle(graph, y, r, v)
        if info["realized_arc_cut"] > best_score + 1e-8:
            best_r, best_labels = next_r.copy(), next_labels.copy()
            best_score = float(info["realized_arc_cut"])
            r = next_r.copy()
            accepted += 1
        trace.append({
            "direction": iteration + 1,
            "mode": mode,
            "arc_max": info["predicted_arc_max"],
            "realized_cut": info["realized_arc_cut"],
            "best_cut": best_score,
            "accepted": bool(info["improvement"] > 1e-8),
            "numerically_valid": info["numerically_valid"],
        })
        if target is not None and best_score >= target:
            break
    final_labels = (y@best_r > 0).astype(np.int8)
    final_score = float(cut_weight(edges, final_labels))
    if (not np.array_equal(final_labels, best_labels)
            or not np.isclose(best_score, final_score, atol=1e-8, rtol=0)
            or final_score+1e-8 < first_score):
        raise AssertionError("Direct rounding lost its original signed Max-Cut")
    return best_r, best_labels, {
        "initial_raw_cut": first_score,
        "optimized_raw_cut": best_score,
        "gain": best_score-first_score,
        "directions_completed": len(trace),
        "improvements": accepted,
        "trace": trace,
        "termination": (
            "target_reached" if target is not None and best_score >= target
            else "direction_limit"
        ),
        "seconds": time.monotonic()-start,
    }
