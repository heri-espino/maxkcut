"""Experiment014: reoptimize saved EVA unit vectors for EXPECTED raw GW cut.

For Y with unit-norm rows and signed undirected weights w_ij:

    E_g[Cut(sign(Y g))] =
        sum_{i<j} w_ij arccos(<y_i,y_j>)/pi.

This is *not* the original EVA linear vector relaxation objective,
nor K-means, nor direct binary vertex flips. We optimize the expected
raw GW rounding score on the product of unit spheres, starting from
the exact original EVA factor. For epsilon>0 we instead optimize
the precisely defined Gaussian-noisy GW expectation

    P[sign(z_i) != sign(z_j)] =
        arccos((1-epsilon)<y_i,y_j>)/pi
    z_i = sqrt(1-epsilon)<y_i,g> + sqrt(epsilon)eta_i,

where the eta_i are independent N(0,1). This makes the objective
smooth near equal vectors and matches the sampling procedure.

Only the *best original-graph expected rounding objective* is retained;
line-search verifies nondecreasing expected value against original Y.
No promise that a particular raw or KL-refined sampled cut improves.
"""
from __future__ import annotations

from functools import lru_cache
import time

import numpy as np

from .io import edge_arrays


@lru_cache(maxsize=1)
def _compiled():
    try:
        from numba import njit
    except ImportError as exc:
        raise ImportError("GW expectation reoptimization requires numba") from exc

    @njit(cache=True)
    def obj_grad(u, v, weights, y, epsilon, want_grad):
        n, p = y.shape
        grad = np.zeros((n, p), dtype=np.float64)
        obj = 0.0
        shrink = 1.0 - epsilon
        for edge in range(len(weights)):
            i, j = u[edge], v[edge]
            dot = 0.0
            for d in range(p):
                dot += y[i, d] * y[j, d]
            corr = shrink * dot
            corr = min(1.0, max(-1.0, corr))
            obj += weights[edge] * np.arccos(corr) / np.pi
            if want_grad:
                # For epsilon=0 exact derivatives are singular when
                # pairs are perfectly aligned. Clip only derivative
                # denominator; accepted steps are checked against
                # the exact, *unclipped* objective above.
                denominator = np.sqrt(max(1e-10, 1.0-corr*corr))
                factor = -weights[edge] * shrink / (np.pi * denominator)
                for d in range(p):
                    grad[i, d] += factor*y[j, d]
                    grad[j, d] += factor*y[i, d]
        if want_grad:
            for i in range(n):
                radial = 0.0
                for d in range(p):
                    radial += grad[i, d]*y[i, d]
                for d in range(p):
                    grad[i, d] -= radial*y[i, d]
        return obj, grad

    return obj_grad


def gw_expected_cut(edges, vectors, *, epsilon=0.):
    """Analytical expectation for iid Gaussian GW rounding of this factor."""
    if not 0 <= epsilon < 1:
        raise ValueError("epsilon must be in [0,1)")
    y = np.asarray(vectors, dtype=np.float64)
    u, v, w = edges
    if y.ndim != 2 or not np.isfinite(y).all():
        raise ValueError("Expected finite vector matrix")
    if np.any(np.abs(np.linalg.norm(y, axis=1)-1.) > 1e-5):
        raise ValueError("Expected unit-length factor rows")
    value, _ = _compiled()(
        np.asarray(u, dtype=np.int64), np.asarray(v, dtype=np.int64),
        np.asarray(w, dtype=np.float64), y, float(epsilon), False,
    )
    return float(value)


def _normalize_rows(a):
    norms = np.linalg.norm(a, axis=1)
    if np.any(norms <= 1e-14) or not np.isfinite(norms).all():
        raise ValueError("Zero/nonfinite row during GW reoptimization")
    return a/norms[:, None]


def optimize_gw_factor(graph, initial, rng, *, epsilon=0.,
                       max_sweeps=60, step=0.08, jitter=0.,
                       patience=12, min_step=1e-7, max_backtracks=10):
    """Projected normalized gradient ascent with strict objective audits.

    Source Y is never mutated. Optionally jitter a seed start on the
    tangent space; the initial SOURCE factor remains the incumbent
    unless a new matrix achieves a greater analytical GW expectation.
    Each accepted step improves the true expected signed Max-Cut,
    not a rounded surrogate. Results contain objective trace, final
    factor and diagnostics. Does not use K-means.
    """
    if (not 0 <= epsilon < 1 or max_sweeps < 1 or step <= 0
            or jitter < 0 or patience < 1 or min_step <= 0
            or max_backtracks < 0):
        raise ValueError("Invalid GW-factor optimizer settings")
    edges = edge_arrays(graph)
    y0 = np.asarray(initial, dtype=np.float64)
    if y0.shape[0] != graph.shape[0] or y0.ndim != 2:
        raise ValueError("Original factor and graph have different sizes")
    if not np.isfinite(y0).all() or np.any(
        np.abs(np.linalg.norm(y0, axis=1)-1.) > 1e-5
    ):
        raise ValueError("Original EVA factor is not finite/unit-normalized")

    kernel = _compiled()
    a, b, weights = edges
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    initial_score, _ = kernel(a, b, weights, y0, epsilon, False)
    y = y0.copy()
    if jitter:
        noise = rng.standard_normal(y.shape)
        noise -= np.einsum("ij,ij->i", noise, y)[:, None]*y
        noise /= np.maximum(np.linalg.norm(noise, axis=1)[:, None], 1e-12)
        y = _normalize_rows(y + jitter*noise)
    current, _ = kernel(a, b, weights, y, epsilon, False)
    best = y0.copy()
    best_score = float(initial_score)
    trace = []
    stagnation = 0
    accepted = 0
    started = time.monotonic()
    reason = "sweep_limit"
    for sweep in range(max_sweeps):
        current, grad = kernel(a, b, weights, y, epsilon, True)
        sizes = np.linalg.norm(grad, axis=1)
        # Normalize tangent gradients row-wise. This retains a
        # strict ascent direction, avoiding very large edge-weight
        # variation across vertices.
        update = grad / np.maximum(sizes[:, None], 1e-12)
        if not np.any(sizes > 1e-10):
            reason = "stationary"
            break
        tried_step = float(step)
        accepted_step = False
        next_score = float(current)
        for backtrack in range(max_backtracks+1):
            candidate = _normalize_rows(y + tried_step*update)
            next_score, _ = kernel(
                a, b, weights, candidate, epsilon, False,
            )
            # Nonmonotone jitter start is permitted but each accepted
            # gradient step must be a STRICT expected-cut ascent.
            if next_score > current + 1e-8:
                y = candidate
                current = next_score
                accepted_step = True
                accepted += 1
                break
            tried_step *= .5
            if tried_step < min_step:
                break
        if not accepted_step:
            stagnation += 1
        elif current > best_score + 1e-8:
            best = y.copy()
            best_score = float(current)
            stagnation = 0
        else:
            stagnation += 1
        trace.append({
            "sweep": sweep+1,
            "candidate_expected": float(current),
            "best_expected": float(best_score),
            "accepted": bool(accepted_step),
            "step": tried_step if accepted_step else None,
            "tangent_gradient_l2": float(np.linalg.norm(grad)),
        })
        if stagnation >= patience:
            reason = "stagnation"
            break
    best_value, _ = kernel(a, b, weights, best, epsilon, False)
    if (best_value + 1e-7 < initial_score
            or not np.isclose(best_value, best_score, atol=1e-6, rtol=0)
            or not np.allclose(np.linalg.norm(best, axis=1), 1., atol=1e-8)):
        raise AssertionError("Expected-cut optimization lost its best factor")
    return best, {
        "initial_expected_cut": float(initial_score),
        "optimized_expected_cut": float(best_value),
        "expected_improvement": float(best_value-initial_score),
        "epsilon": float(epsilon),
        "jitter": float(jitter),
        "step": float(step),
        "max_sweeps": int(max_sweeps),
        "sweeps_done": len(trace),
        "gradient_steps_accepted": int(accepted),
        "termination": reason,
        "trace": trace,
        "seconds": time.monotonic()-started,
    }


def raw_gw_pair(before, after, rng, *, epsilon=0.):
    """Coupled Gaussian hyperplane rounding on old/new factors.

    Same Gaussian direction and same optional per-vertex Gaussian
    noise ensure a matched/paired before-after comparison.
    For epsilon=0 this is ordinary raw GW (no clustering/extra noise).
    """
    if before.shape != after.shape:
        raise ValueError("GW factors must have identical dimensions")
    if not 0 <= epsilon < 1:
        raise ValueError("Invalid raw rounding epsilon")
    g = rng.standard_normal(before.shape[1])
    eta = rng.standard_normal(before.shape[0]) if epsilon else 0.
    scale = np.sqrt(1.-epsilon)
    magnitude = np.sqrt(epsilon)
    source = (scale*(before@g) + magnitude*eta >= 0).astype(np.int8)
    refined = (scale*(after@g) + magnitude*eta >= 0).astype(np.int8)
    return source, refined
