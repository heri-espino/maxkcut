"""Objective-aware spherical K-MeansNM rounding for signed weighted Max-Cut.

For k=2, nearest-centroid spherical K-Means is a HOMOGENEOUS hyperplane
split: label_i = 1[(Y_i) @ (c_1-c_0) >= 0]. Unlike clustering inertia,
our objective is the ORIGINAL signed G-set weighted Max-Cut.

This experiment first creates the exact same seeded K-MeansNM direction
trajectory as src/maxkcut/rounding.py, retaining the best original cut.
For EVERY direction it then finds the globally optimal separating
threshold along that one projection in O(n log n + m) arithmetic:
sort projected vertices and sweep signed vertex flip gains. No dense
n x n matrix, no EVA rerun, no PCA.

After full-dimensional K-Means directions, explore objective-scored
nearby hyperplanes with Gaussian, centroid-recalibrated and
signed-graph gain-weighted proposals. Always keep the incumbent
highest weighted Max-Cut. Finally the caller can refine using
the existing compiled generalized KL optimizer, which is NOT
restricted to linearly separable EVA cuts.

This is a heuristic over DIRECTIONS; only the scalar threshold is
exactly optimized conditional on each fixed direction.
"""
from __future__ import annotations

from functools import lru_cache
import time

import numpy as np

from .io import edge_arrays
from .objective import cut_weight


@lru_cache(maxsize=1)
def _sweep_kernel():
    try:
        from numba import njit
    except ImportError as exc:
        raise ImportError(
            'Exact threshold sweeps require Numba; install '
            'python -m pip install "numba>=0.67,<0.69"'
        ) from exc

    @njit(cache=True)
    def sweep(indptr, indices, weights, order, projection):
        """Move equal-projection values together; exact signed flip gains.

        Initially every vertex has label 0, giving cut zero.
        Moving a 0->1 vertex changes the cut by sum(weights to label0)
        minus sum(weights to label1). Each edge gain update is constant
        work when one of its endpoints moves.

        Return the best prefix LENGTH (0..n), highest exact cut score;
        ties retain smallest prefix length.
        """
        n = len(order)
        gains = np.zeros(n, dtype=np.float64)
        for i in range(n):
            value = 0.0
            for ptr in range(indptr[i], indptr[i + 1]):
                value += weights[ptr]
            gains[i] = value
        best_score = 0.0
        best_prefix = 0
        current = 0.0
        prefix = 0
        while prefix < n:
            begin = prefix
            pivot = projection[order[prefix]]
            while prefix < n and projection[order[prefix]] == pivot:
                vertex = order[prefix]
                current += gains[vertex]
                for ptr in range(indptr[vertex], indptr[vertex + 1]):
                    gains[indices[ptr]] -= 2.0 * weights[ptr]
                prefix += 1
            # Only legal separators between distinct projected values,
            # or all-same partitions, not within tied coordinates.
            if current > best_score + 1e-10:
                best_score = current
                best_prefix = prefix
        return best_prefix, best_score

    return sweep


def optimal_threshold(adjacency, projections):
    """Exact best signed Max-Cut threshold for one n-vertex projection.

    Returns 0/1 partition and report. Among score ties, choose a
    smallest sorted prefix (including the all-equal no-cut partition).
    The partition can be unbalanced; we do not enforce class sizes.
    """
    z = np.asarray(projections, dtype=np.float64)
    n = adjacency.shape[0]
    if z.shape != (n,) or not np.isfinite(z).all() or n < 2:
        raise ValueError("Projection must be finite and match graph vertices")
    order = np.argsort(z, kind="stable").astype(np.int32)
    indptr = np.asarray(adjacency.indptr)
    indices = np.asarray(adjacency.indices)
    weights = np.asarray(adjacency.data, dtype=np.float64)
    prefix, score = _sweep_kernel()(
        indptr, indices, weights, order, z,
    )
    labels = np.zeros(n, dtype=np.int8)
    labels[order[:prefix]] = 1
    actual = float(cut_weight(edge_arrays(adjacency), labels))
    if not np.isclose(actual, score, atol=1e-7, rtol=0):
        raise RuntimeError("Threshold incremental cut mismatches original graph")
    if prefix == 0:
        threshold = float("-inf")
    elif prefix == n:
        threshold = float("inf")
    else:
        # Midpoint can overflow/underflow for extreme projections;
        # the integer prefix is the authoritative partition.
        threshold = float(z[order[prefix - 1]] / 2 + z[order[prefix]] / 2)
    return labels, {
        "cut": actual, "sorted_prefix": int(prefix),
        "threshold": threshold if np.isfinite(threshold) else None,
        "group_sizes": [int(n-prefix), int(prefix)],
        "distinct_projection_values": int(np.unique(z).size),
    }


def _unit(vec):
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-14:
        return None
    return vec / norm


def _centroid_direction(vectors, labels):
    mask = np.asarray(labels, dtype=np.int8) == 1
    if not mask.any() or mask.all():
        return None
    # Spherical K-Means means, normalized per cluster.
    a = _unit(np.mean(vectors[~mask], axis=0))
    b = _unit(np.mean(vectors[mask], axis=0))
    if a is None or b is None:
        return None
    return _unit(b-a)


def _weighted_direction(vectors, adjacency, labels):
    """Cut-aware continuous proxy: emphasize vertices with flip incentive.

    This is a direction PROPOSAL, not the gradient of discrete Max-Cut.
    Signed graph local flip gains identify boundary labels whose
    hyperplane side is most questionable; small-gain vertices receive
    higher positive weight to steer the separating normal.
    """
    spin = (1 - 2*np.asarray(labels, dtype=np.int8)).astype(np.float64)
    gains = spin * np.asarray(adjacency @ spin).ravel()
    weights = 1.0/(1.0 + np.abs(gains))
    signed = (2*np.asarray(labels, dtype=np.int8) - 1).astype(np.float64)
    return _unit(vectors.T @ (signed*weights))


def improve_kmeans_direction(
    vectors, graph, rng, *, direction_trials=96,
    kmeans_iters=50, perturb_scales=(0.03,0.08,0.16,0.35),
    threshold_search=True, target=None,
):
    """Best original signed graph cut across K-MeansNM and direction trials.

    Reproduces EXACTLY the former randomized antipodal 2-center
    initialization and Lloyd centroid updates (including random
    singleton/empty-center fallback) for the baseline trajectory.
    Keeping the best of all original zero-threshold Lloyd assignments
    is identical to round_spherical_kmeans(...,k=2,antipodal=True,
    keep_best=True) under the same seed when kmeans_iters=100.

    All new threshold and perturbation candidates use the same
    original weighted Max-Cut, not clustering inertia. A direction
    conditional-threshold optimum is exact; searching all possible
    directions is heuristic and NOT globally optimal.
    """
    y = np.asarray(vectors, dtype=np.float64)
    if (y.ndim != 2 or y.shape[0] != graph.shape[0]
            or y.shape[0] < 2 or y.shape[1] < 1
            or not np.isfinite(y).all()):
        raise ValueError("Expected finite full-dimensional EVA vectors")
    if (direction_trials < 0 or kmeans_iters < 1
            or not perturb_scales
            or any(not np.isfinite(x) or x <= 0 for x in perturb_scales)):
        raise ValueError("Invalid K-Means direction search settings")
    edges = edge_arrays(graph)
    n, p = y.shape
    started = time.monotonic()
    centers = rng.standard_normal((2, p))
    centers[1] = -centers[0]
    centers /= np.linalg.norm(centers, axis=1)[:, None]
    best_labels = None
    best_cut = float("-inf")
    best_direction = None
    best_threshold = None
    best_source = None
    zero_best = float("-inf")
    zero_labels = None
    zero_iters = 0
    evaluated = 0
    improvement_trace = []
    previous = None

    def consider(direction, origin):
        nonlocal best_cut, best_labels, best_direction, best_threshold
        nonlocal evaluated, best_source, zero_best, zero_labels
        direction = _unit(direction)
        if direction is None:
            return False
        z = y @ direction
        zero_partition = (z > 0).astype(np.int8)
        score_zero = float(cut_weight(edges, zero_partition))
        if score_zero > zero_best:
            zero_best = score_zero
            zero_labels = zero_partition.copy()
        if threshold_search:
            part, diag = optimal_threshold(graph, z)
        else:
            part = zero_partition
            diag = {"cut": score_zero, "threshold": 0.0}
        score = float(diag["cut"])
        evaluated += 1
        if score > best_cut + 1e-10:
            best_cut = score
            best_labels = np.asarray(part, dtype=np.int8)
            best_direction = direction.copy()
            best_threshold = diag["threshold"]
            best_source = origin
            improvement_trace.append({
                "candidate": evaluated, "stage": origin,
                "best_cut": score,
            })
            return True
        return False

    # Original K-MeansNM trajectory and its initial random hyperplane.
    # Identical raw assignments to the source implementation's
    # np.argmax(vectors @ centers.T) tie semantics.
    for iteration in range(1, kmeans_iters + 1):
        labels = np.argmax(y @ centers.T, axis=1)
        direction = centers[1] - centers[0]
        # Track original labels explicitly for identical ties.
        original_score = float(cut_weight(edges, labels))
        if original_score > zero_best:
            zero_best = original_score
            zero_labels = labels.astype(np.int8).copy()
        consider(direction, f"kmeans-iteration-{iteration}")
        zero_iters = iteration
        if previous is not None and np.array_equal(previous, labels):
            break
        previous = labels.copy()
        for c in range(2):
            members = y[labels == c]
            if not len(members):
                centers[c] = y[rng.integers(len(y))]
            else:
                center = np.sum(members, axis=0)
                norm = np.linalg.norm(center)
                centers[c] = (
                    center/norm if norm > 1e-12
                    else y[rng.integers(len(y))]
                )

    # Perturb BOTH strong near-centroid directions and successful
    # incumbent hyperplanes, not only random Gaussian proposals.
    for trial in range(direction_trials):
        if best_cut >= target if target is not None else False:
            break
        action = trial % 6
        if action in (0, 1, 2, 3):
            base = (
                best_direction if rng.random() < 0.8
                else _unit(rng.standard_normal(p))
            )
            noise = _unit(rng.standard_normal(p))
            if base is None or noise is None:
                continue
            scale = perturb_scales[trial % len(perturb_scales)]
            candidate = base + scale*noise
        elif action == 4:
            candidate = _centroid_direction(y, best_labels)
        else:
            candidate = _weighted_direction(y, graph, best_labels)
        consider(candidate, f"direction-trial-{trial}-{action}")
    final_score = float(cut_weight(edges, best_labels))
    if (not np.isclose(final_score, best_cut, rtol=0, atol=1e-8)
            or final_score + 1e-8 < zero_best):
        raise RuntimeError("Cut-aware search lost the original K-Means baseline")
    return best_labels, {
        "cut": final_score,
        "raw_kmeans_best_cut": zero_best,
        "raw_kmeans_labels": zero_labels,
        "kmeans_iterations": zero_iters,
        "direction_candidates": evaluated,
        "direction_trials": direction_trials,
        "best_source": best_source,
        "best_threshold": best_threshold,
        "improvement_trace": improvement_trace,
        "seconds": time.monotonic() - started,
        "uses_full_eva_dimensions": True,
        "conditional_threshold_exact": bool(threshold_search),
        "objective": "original_signed_weighted_max_cut",
    }
