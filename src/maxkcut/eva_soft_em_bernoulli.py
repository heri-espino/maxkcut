"""Experiment016: soft spherical EM / Bernoulli search for signed Max-Cut.

EVA Y remains IMMUTABLE. There is no new EVA solve and no K-means.
Soft alternating mean-field/centroid updates optimize a consistent
entropy-regularized objective for each fixed temperature:

  L(p,c0,c1;T) =
    sum_{i<j} w_ij (p_i+p_j-2*p_i*p_j)
    + kappa sum_i [(1-p_i) <Y_i,c0>+p_i <Y_i,c1>]
    + T sum_i H(p_i), ||c0||=||c1||=1.

Conditional coordinate optimum is
  p_i = sigmoid((sum_j w_ij*(1-2p_j) +
                  kappa*Y_i@(c1-c0))/T).

We perform damped synchronous mean-field steps with objective-audited
backtracking, followed by exact unit-sphere centroid maximization.
Annealing changes the objective, so monotonicity only applies within
an individual temperature stage.

The output is a vector of Bernoulli probabilities, not a hard
assignment. CEM then fits an independent Bernoulli search law to
complement-aligned high-scoring partitions, while optional Gibbs
annealing samples *conditional* probabilities using the ORIGINAL
signed graph. Both retain the best hard feasible cut. KL is an
optional, independently audited postprocessor, not part of EM.
"""
from __future__ import annotations

from functools import lru_cache
import time

import numpy as np
from scipy.special import expit

from .io import edge_arrays
from .objective import cut_weight


def _unit(x):
    x = np.asarray(x, dtype=np.float64)
    length = float(np.linalg.norm(x))
    return None if not np.isfinite(length) or length < 1e-12 else x / length


def _free_energy(y, edges, p, c0, c1, kappa, temperature):
    i, j, w = edges
    cut = float(np.dot(w, p[i] + p[j] - 2*p[i]*p[j]))
    geometry = kappa * float(
        np.dot(1-p, y@c0) + np.dot(p, y@c1)
    )
    safe = np.clip(p, 1e-12, 1-1e-12)
    entropy = float(np.sum(
        -safe*np.log(safe)-(1-safe)*np.log1p(-safe)
    ))
    return cut+geometry+temperature*entropy


def _centroids(y, p, c0, c1):
    a = _unit((1-p)@y)
    b = _unit(p@y)
    return (c0 if a is None else a), (c1 if b is None else b)


def soft_em(graph, y, rng, *, mode="mean", anchor=None, kappa=1.,
            delta=.25, temp_start=5., temp_end=.15, stages=8,
            em_iters=20, damping=.4):
    """Seeded soft EM with audited mean-field/centroid coordinate ascent.

    'mean' starts from perturbed mean direction on the sphere;
    'elite' additionally starts Bernoulli memberships near a verified
    existing strong binary partition. No hard reclassification
    during EM; outputs remain in [0,1].
    """
    if mode not in ("mean", "elite"):
        raise ValueError("Unknown soft EM initialization")
    y = np.asarray(y, dtype=np.float64)
    n, p_dim = y.shape
    if (graph.shape != (n,n) or p_dim < 2 or not np.isfinite(y).all()
            or kappa < 0 or delta <= 0 or temp_start <= 0
            or temp_end <= 0 or stages < 1 or em_iters < 1
            or not 0 < damping <= 1):
        raise ValueError("Invalid soft EM graph/factor/budget")
    if np.any(np.abs(np.linalg.norm(y,axis=1)-1)>1e-5):
        raise ValueError("EVA vectors must be normalized")
    if mode == "elite" and (
        anchor is None or np.asarray(anchor).shape!=(n,)
        or not np.isin(anchor,[0,1]).all()
    ):
        raise ValueError("Elite soft EM requires verified binary labels")

    m = _unit(y.mean(axis=0))
    if m is None:
        m = _unit(rng.standard_normal(p_dim))
    v = rng.standard_normal(p_dim)
    v -= (v@m)*m
    v = _unit(v)
    if v is None:
        raise ValueError("Cannot obtain spherical centroid perturbation")
    c0 = _unit(m-delta*v)
    c1 = _unit(m+delta*v)
    if mode == "mean":
        p = np.clip(
            expit(kappa*(y@(c1-c0))/temp_start),1e-7,1-1e-7
        )
    else:
        # High-quality graph solution used as an informed probability
        # prior, without freezing or fixing any vertex assignment.
        p = np.where(np.asarray(anchor,dtype=bool),.95,.05)
        c0,c1 = _centroids(y,p,c0,c1)
    edges = edge_arrays(graph)
    temperatures=np.geomspace(temp_start,temp_end,stages)
    trace=[]
    began=time.monotonic()
    for stage, temperature in enumerate(temperatures):
        last=_free_energy(y,edges,p,c0,c1,kappa,temperature)
        for iteration in range(em_iters):
            field=np.asarray(graph@(1.-2.*p)).ravel()
            target=expit(
                (field+kappa*(y@(c1-c0)))/temperature
            )
            rate=float(damping)
            proposed=p.copy()
            score=last
            accepted=False
            for _ in range(14):
                proposed=np.clip(
                    (1-rate)*p+rate*target,1e-7,1-1e-7
                )
                score=_free_energy(
                    y,edges,proposed,c0,c1,kappa,temperature
                )
                if score>=last-1e-8:
                    accepted=True
                    break
                rate*=.5
            if accepted:
                p=proposed
                last=score
            c0,c1=_centroids(y,p,c0,c1)
            after=_free_energy(y,edges,p,c0,c1,kappa,temperature)
            if after+1e-6<last:
                raise RuntimeError("Soft EM centroid M-step decreased its free energy")
            change=abs(after-last)
            last=after
            trace.append({
                "stage":stage+1, "iteration":iteration+1,
                "temperature":float(temperature),
                "free_energy":float(after),
                "accepted_E_step":accepted,
                "mean_bernoulli":float(np.mean(p)),
                "mean_entropy":float(np.mean(
                    -p*np.log(p)-(1-p)*np.log1p(-p)
                )),
            })
            if not accepted and change < 1e-8:
                break
    labels=(p>=.5).astype(np.int8)
    return p, labels, {
        "mode":mode,"trace":trace,
        "hard_cut":float(cut_weight(edges,labels)),
        "seconds":time.monotonic()-began,
    }


def aligned(reference, candidate):
    """Global complement symmetry: x and 1-x represent the SAME cut."""
    reference=np.asarray(reference,dtype=np.int8)
    x=np.asarray(candidate,dtype=np.int8)
    if np.count_nonzero(reference!=x)>len(x)//2:
        return 1-x
    return x.copy()


def _samples(edges, probabilities, rng, count):
    """Uniform-width independent Bernoulli samples scored on signed graph."""
    samples=(rng.random((count,len(probabilities)))<
             probabilities).astype(np.int8)
    u,v,w=edges
    # Avoid an enormous (population x m) temporary when large.
    values=np.zeros(count,dtype=float)
    for start in range(0,len(u),8192):
        stop=min(start+8192,len(u))
        values+=np.not_equal(
            samples[:,u[start:stop]],
            samples[:,v[start:stop]],
        ) @ w[start:stop]
    return samples,values


def cem(graph, initial_p, rng, *, anchor=None,
        generations=30, population=96, elite_fraction=.15,
        update_rate=.6, min_probability=.003,
        anchor_weight=.96, injection=.035, target=None):
    """Complement-aligned elite CEM on independent Bernoulli partitions.

    Strong anchor conditioning makes independent Bernoulli sampling
    viable in thousands of dimensions. The best *hard* graph cut is
    always retained; CEM does not impose monotone sample scores.
    """
    p=np.asarray(initial_p,dtype=float).copy()
    n=len(p)
    if (p.ndim!=1 or graph.shape!=(n,n) or not np.isfinite(p).all()
            or np.any((p<0)|(p>1)) or generations<1 or population<2
            or not 0<elite_fraction<=1 or not 0<update_rate<=1
            or not 0<min_probability<.5 or not 0<=anchor_weight<=1
            or not 0<=injection<.5):
        raise ValueError("Invalid CEM probabilities/hyperparameters")
    if anchor is not None and (
        np.asarray(anchor).shape!=(n,)
        or not np.isin(anchor,[0,1]).all()
    ):
        raise ValueError("Invalid CEM anchor")
    edges=edge_arrays(graph)
    starting=(p>=.5).astype(np.int8)
    if anchor is not None:
        anchor=np.asarray(anchor,dtype=np.int8)
        if float(cut_weight(edges,anchor)) >= float(
            cut_weight(edges,starting)
        ):
            best=anchor.copy()
        else:
            best=starting.copy()
        anchor=aligned(best,anchor)
        p=(1-anchor_weight)*p+anchor_weight*anchor.astype(float)
    else:
        best=starting.copy()
    p=np.clip(p,min_probability,1-min_probability)
    best_score=float(cut_weight(edges,best))
    first=best_score
    count=max(1,int(np.ceil(population*elite_fraction)))
    trace=[]
    began=time.monotonic()
    for generation in range(generations):
        samples,scores=_samples(edges,p,rng,population)
        best_index=int(np.argmax(scores))
        if scores[best_index]>best_score+1e-8:
            best=samples[best_index].copy()
            best_score=float(scores[best_index])
        ranking=np.argsort(-scores,kind="stable")[:count]
        elite=np.stack([
            aligned(best,samples[idx]) for idx in ranking
        ]).astype(float)
        fitted=elite.mean(axis=0)
        p=(1-update_rate)*p+update_rate*fitted
        # Keep nonzero exploration, without erasing the learned
        # elite marginal probabilities on all coordinates.
        if injection:
            p=(1-injection)*p+injection*.5
        p=np.clip(p,min_probability,1-min_probability)
        trace.append({
            "generation":generation+1,
            "best_cut":best_score,
            "population_best":float(np.max(scores)),
            "population_mean":float(np.mean(scores)),
            "mean_entropy":float(np.mean(
                -p*np.log(p)-(1-p)*np.log1p(-p)
            )),
        })
        if target is not None and best_score>=target:
            break
    verified=float(cut_weight(edges,best))
    if not np.isclose(verified,best_score,rtol=0,atol=1e-7):
        raise AssertionError("CEM lost its best original-graph binary cut")
    return p,best,{
        "initial_best_cut":first,
        "best_cut":best_score,
        "improvement":best_score-first,
        "trace":trace,
        "seconds":time.monotonic()-began,
    }


@lru_cache(maxsize=1)
def _gibbs_kernel():
    from numba import njit

    @njit(cache=True)
    def sweep(indptr,indices,weights,spins,ordering,uniforms,temp):
        gain_total=0.
        for pos in range(len(ordering)):
            i=ordering[pos]
            field=0.
            for ptr in range(indptr[i],indptr[i+1]):
                field+=weights[ptr]*spins[indices[ptr]]
            gain=spins[i]*field
            q=1./(1.+np.exp(-max(-60.,min(60.,gain/temp))))
            if uniforms[pos]<q:
                spins[i]=-spins[i]
                gain_total+=gain
        return gain_total

    return sweep


def gibbs_anneal(graph, initial, rng, *, sweeps=150,
                 temp_start=12.,temp_end=.4,target=None):
    """Single-site Gibbs heat bath on ORIGINAL signed Max-Cut energy.

    Exact graph flip gain is computed with current neighboring spins;
    conditional Bernoulli update samples the opposite current spin
    with probability sigmoid(gain/temperature). Anneal geometrically.
    The CURRENT Markov state can worsen; best hard cut is retained.
    """
    y=np.asarray(initial,dtype=np.int8)
    n=len(y)
    if (y.ndim!=1 or graph.shape!=(n,n)
            or not np.isin(y,[0,1]).all()
            or sweeps<1 or temp_start<=0 or temp_end<=0):
        raise ValueError("Invalid Gibbs seed/temperature")
    a=graph.tocsr()
    a.sort_indices()
    spins=1.-2.*y.astype(np.float64)
    edges=edge_arrays(a)
    score=float(cut_weight(edges,y))
    best=y.copy()
    best_score=score
    first=score
    started=time.monotonic()
    trace=[]
    kernel=_gibbs_kernel()
    for t,temp in enumerate(np.geomspace(temp_start,temp_end,sweeps)):
        order=np.asarray(rng.permutation(n),dtype=np.int32)
        delta=kernel(a.indptr,a.indices,a.data,spins,order,
                     rng.random(n),float(temp))
        labels=(spins<0).astype(np.int8)
        current=float(cut_weight(edges,labels))
        if not np.isclose(score+delta,current,rtol=0,atol=1e-6):
            raise AssertionError("Gibbs incremental gain disagrees with signed Max-Cut")
        score=current
        if score>best_score+1e-8:
            best_score=score
            best=labels.copy()
        trace.append({
            "sweep":t+1, "temperature":float(temp),
            "current_cut":score, "best_cut":best_score,
        })
        if target is not None and best_score>=target:
            break
    return best,{
        "initial_cut":first, "best_cut":best_score,
        "trace":trace, "seconds":time.monotonic()-started,
    }
