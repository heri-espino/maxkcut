"""Stochastic optimization of GW rounding NORMALS with frozen EVA vectors.

ES-SGA maximizes a Gaussian-smoothed version of the ACTUAL discontinuous,
signed, weighted, original-graph Max-Cut of raw hyperplane rounding:

    F_h(r) = E_eps[C(sign(Y(r+h*eps)))]
    grad F_h(r) = E[(C(r+h*eps)-C(r-h*eps))*eps]/(2*h)

The paired antithetic estimator is unbiased for this Euclidean smooth
objective; each stochastic step projects the estimate to the tangent
space of the unit direction r. h = angular_noise / sqrt(p). Adam
updates are renormalized; only the BEST observed HARD cut is returned,
including every sampled antithetic direction.

Soft-SGA is a second, distinct signed-edge-minibatch baseline. It
maximizes the differentiable, independently Bernoulli-rounded surrogate

    E[cut] = sum_e w_ij * (p_i+p_j-2*p_i*p_j),
    p_i = sigmoid((Y_i @ r)/temperature).

Uniformly drawn signed-edge minibatches give an unbiased gradient of
THAT relaxed objective, not of the hard deterministic cut or the shared
Gaussian-hyperplane GW distribution. Temperatures anneal; every
completed update is also assessed with the actual hard Max-Cut.
Neither solver changes Y or guarantees a global optimum.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.special import expit

from .io import edge_arrays
from .objective import cut_weight


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= 1e-13:
        return None
    return vector/length


def _validate(graph, vectors, initial, steps, learning_rate):
    y = np.asarray(vectors, dtype=np.float64)
    r = _unit(initial)
    if (y.ndim != 2 or y.shape[0] != graph.shape[0]
            or r is None or y.shape[1] != len(r)
            or not np.isfinite(y).all()
            or steps < 1 or not np.isfinite(learning_rate)
            or learning_rate <= 0):
        raise ValueError("Invalid frozen EVA vectors, normal or stochastic budget")
    return y, r, edge_arrays(graph)


def _score_batch(edges, projections):
    """Signed original-graph objective for B real hyperplane candidates."""
    u, v, w = edges
    columns = np.asarray(projections, dtype=np.float64)
    if columns.ndim != 2:
        raise ValueError("Expected (n,candidates) projection matrix")
    signs = columns > 0
    # A dense (m, B) boolean matrix is much smaller than (n, n);
    # no dense adjacency or Gram matrix is constructed.
    return np.asarray(
        w @ np.not_equal(signs[u], signs[v]),
        dtype=np.float64,
    )


def _adam_step(r, grad, momentum, velocity, k, learning_rate):
    if not np.isfinite(grad).all():
        raise RuntimeError("Nonfinite stochastic rounding gradient")
    grad = grad - float(grad@r)*r
    momentum = .9*momentum+.1*grad
    velocity = .999*velocity+.001*(grad*grad)
    mh = momentum/(1.-.9**k)
    vh = velocity/(1.-.999**k)
    update = mh/(np.sqrt(vh)+1e-8)
    update -= float(update@r)*r
    length = float(np.linalg.norm(update))
    if length <= 1e-13:
        return r.copy(), momentum, velocity
    next_r = _unit(r+learning_rate*(update/length))
    if next_r is None:
        raise RuntimeError("SGA produced invalid direction")
    return next_r, momentum, velocity


def _noise_schedule(begin, end, index, count):
    # Geometric annealing handles scales spanning multiple orders.
    return float(begin * (end/begin)**(index/max(1,count-1)))


def optimize_es_sga(
    graph, vectors, rng, *, initial, steps=300, population=16,
    sigma_start=.3, sigma_end=.025, learning_rate=.08,
    restart_patience=80, target=None,
):
    """Antithetic Gaussian-smoothed SGD/Adam over raw GW normals.

    SGD samples *actual graph cuts*, not K-means and not a soft
    logistic proxy. No accepted current-point step is required to
    improve, but the best hard feasible cut is never discarded.
    All sample directions (r+h*eps and r-h*eps) are eligible incumbents.
    """
    y, current, edges = _validate(
        graph, vectors, initial, steps, learning_rate,
    )
    if (population < 1 or sigma_start <= 0 or sigma_end <= 0
            or not np.isfinite(sigma_start+sigma_end)
            or restart_patience < 1):
        raise ValueError("Invalid antithetic population/noise settings")
    graph_size = y.shape[1]
    initial_labels = (y@current > 0).astype(np.int8)
    initial_score = float(cut_weight(edges, initial_labels))
    best = current.copy()
    best_score = initial_score
    best_labels = initial_labels.copy()
    momentum = np.zeros_like(current)
    velocity = np.zeros_like(current)
    trace = []
    improvements = 0
    resets = 0
    last_gain_step = 0
    began = time.monotonic()

    def consider(candidate, measured_score):
        nonlocal best, best_labels, best_score, improvements, last_gain_step
        if measured_score <= best_score + 1e-8:
            return
        normalized = _unit(candidate)
        if normalized is None:
            return
        labels = (y@normalized > 0).astype(np.int8)
        cut = float(cut_weight(edges, labels))
        if not np.isclose(cut, measured_score, rtol=0, atol=1e-7):
            # numerical zero-boundary anomaly; retain only real cuts
            measured_score = cut
        if measured_score > best_score + 1e-8:
            best = normalized
            best_labels = labels
            best_score = float(measured_score)
            improvements += 1
            last_gain_step = len(trace)+1

    for iteration in range(steps):
        noise_strength = _noise_schedule(
            sigma_start, sigma_end, iteration, steps,
        )
        h = noise_strength/np.sqrt(graph_size)
        eps = rng.standard_normal((population, graph_size))
        plus = current[None, :]+h*eps
        minus = current[None, :]-h*eps
        # Matrix multiplication uses BLAS for all perturbations in one
        # operation; graph scoring is vectorized over undirected edges.
        candidate_normals = np.vstack((plus, minus)).T
        samples = _score_batch(edges, y@candidate_normals)
        diff = samples[:population]-samples[population:]
        grad = (diff@eps)/(2.*h*population)
        # For the Gaussian-smoothed EUCLIDEAN function, this antithetic
        # estimator is unbiased (sampling noise remains finite).
        best_probe = int(np.argmax(samples))
        if samples[best_probe] > best_score+1e-8:
            consider(candidate_normals[:,best_probe], samples[best_probe])
        updated, momentum, velocity = _adam_step(
            current, grad, momentum, velocity,
            iteration+1, learning_rate,
        )
        current = updated
        current_score = float(cut_weight(
            edges, (y@current > 0).astype(np.int8),
        ))
        if current_score > best_score+1e-8:
            consider(current, current_score)
        if iteration+1-last_gain_step >= restart_patience:
            # Return to the elite neighborhood but randomize the
            # current state slightly; best-so-far remains untouched.
            tangent = rng.standard_normal(graph_size)
            tangent -= float(tangent@best)*best
            tangent = _unit(tangent)
            current = best.copy() if tangent is None else _unit(
                best+noise_strength*tangent
            )
            momentum.fill(0)
            velocity.fill(0)
            resets += 1
            last_gain_step = iteration+1
        trace.append({
            "step": iteration+1,
            "noise": noise_strength,
            "probe_max": float(np.max(samples)),
            "candidate_cut": current_score,
            "best_cut": best_score,
            "estimated_gradient_norm": float(np.linalg.norm(grad)),
        })
        if target is not None and best_score >= target:
            break
    final_labels = (y@best > 0).astype(np.int8)
    final_score = float(cut_weight(edges, final_labels))
    if (final_score+1e-8 < initial_score
            or not np.isclose(final_score, best_score, atol=1e-8, rtol=0)
            or not np.array_equal(final_labels, best_labels)):
        raise AssertionError("ES-SGA lost its best actual weighted Max-Cut")
    return best, final_labels, {
        "optimizer": "es-sga",
        "initial_raw_cut": initial_score,
        "optimized_raw_cut": final_score,
        "gain": final_score-initial_score,
        "steps_completed": len(trace),
        "improvements": improvements,
        "basin_resets": resets,
        "trace": trace,
        "termination": "target_reached" if target is not None and final_score>=target else "step_limit",
        "seconds": time.monotonic()-began,
    }


def optimize_soft_sga(
    graph, vectors, rng, *, initial, steps=300,
    edge_batch=2048, temperature_start=.5, temperature_end=.04,
    learning_rate=.08, restart_patience=80, target=None,
):
    """Signed-edge minibatch Adam on smooth independently-rounded cut.

    The gradient is unbiased with respect to the SOFT surrogate, not
    the original hard hyperplane cut. Only hard rounded actual original
    graph scores determine the stored best feasible solution.
    """
    y, current, edges = _validate(
        graph, vectors, initial, steps, learning_rate,
    )
    u, v, w = edges
    if (edge_batch < 1 or temperature_start <= 0
            or temperature_end <= 0
            or not np.isfinite(temperature_start+temperature_end)
            or restart_patience < 1):
        raise ValueError("Invalid soft SGD edge batch/temperatures")
    initial_labels = (y@current > 0).astype(np.int8)
    initial_score = float(cut_weight(edges, initial_labels))
    best = current.copy()
    best_labels = initial_labels.copy()
    best_score = initial_score
    momentum = np.zeros_like(current)
    velocity = np.zeros_like(current)
    improvements, resets, last_gain_step = 0, 0, 0
    began = time.monotonic()
    trace = []
    m = len(w)
    for k in range(steps):
        temperature = _noise_schedule(
            temperature_start, temperature_end, k, steps,
        )
        prob = expit((y@current)/temperature)
        indices = rng.integers(m, size=edge_batch)
        ii, jj, ww = u[indices], v[indices], w[indices]
        # d/dz_i [p_i+p_j-2*p_i*p_j] =
        #   p_i*(1-p_i)*(1-2*p_j) / temperature.
        vi = (m/edge_batch) * ww * (
            prob[ii]*(1.-prob[ii])*(1.-2.*prob[jj])/temperature
        )
        vj = (m/edge_batch) * ww * (
            prob[jj]*(1.-prob[jj])*(1.-2.*prob[ii])/temperature
        )
        coefficients = (
            np.bincount(ii, weights=vi, minlength=y.shape[0])+
            np.bincount(jj, weights=vj, minlength=y.shape[0])
        )
        grad = y.T @ coefficients
        current, momentum, velocity = _adam_step(
            current, grad, momentum, velocity,
            k+1, learning_rate,
        )
        candidate_labels = (y@current > 0).astype(np.int8)
        current_score = float(cut_weight(edges, candidate_labels))
        if current_score > best_score + 1e-8:
            best = current.copy()
            best_labels = candidate_labels.copy()
            best_score = current_score
            improvements += 1
            last_gain_step = k+1
        if k+1-last_gain_step >= restart_patience:
            noise = rng.standard_normal(current.size)
            noise -= float(noise@best)*best
            noise = _unit(noise)
            current = best.copy() if noise is None else _unit(
                best + 4.*temperature*noise
            )
            momentum.fill(0)
            velocity.fill(0)
            resets += 1
            last_gain_step = k+1
        trace.append({
            "step": k+1, "temperature": temperature,
            "candidate_cut": current_score,
            "best_cut": best_score,
            "estimated_gradient_norm": float(np.linalg.norm(grad)),
        })
        if target is not None and best_score >= target:
            break
    actual = float(cut_weight(edges, (y@best > 0).astype(np.int8)))
    if (actual+1e-8 < initial_score or
            not np.isclose(actual, best_score, rtol=0, atol=1e-8)
            or not np.array_equal((y@best > 0).astype(np.int8), best_labels)):
        raise AssertionError("Soft-SGA lost its best actual weighted Max-Cut")
    return best, best_labels, {
        "optimizer": "soft-sga",
        "initial_raw_cut": initial_score,
        "optimized_raw_cut": best_score,
        "gain": best_score-initial_score,
        "steps_completed": len(trace),
        "improvements": improvements,
        "basin_resets": resets,
        "trace": trace,
        "termination": "target_reached" if target is not None and best_score>=target else "step_limit",
        "seconds": time.monotonic()-began,
    }
