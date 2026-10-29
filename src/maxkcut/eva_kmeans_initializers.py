"""Experiment018: alternative initial spherical K-Means centroids for k=2.

The original two-centroid antipodal Gaussian K-MeansNM ('paired') is
unchanged and remains the same-seed control. All new modes only
initialize (c0,c1); the subsequent Lloyd updates and original signed
G-set cut evaluation remain exactly the same as Experiment017.

'gw-screen16' is a declared *cut-aware* initializer: it spends 16
original-graph hard-cut evaluations before Lloyd, so its compute
budget differs from purely geometric initializers. This distinction
is recorded in experiment metadata and comparisons.
"""
from __future__ import annotations

import numpy as np

from .objective import cut_weight

INIT2=(
    "paired", "independent", "sample-antipodal", "sample-pair",
    "farthest-pair", "spherical-pp", "mean-perturb",
    "pca-antipodal", "gw-screen16",
)


def _unit(x):
    x=np.asarray(x,dtype=np.float64)
    norm=float(np.linalg.norm(x))
    return None if not np.isfinite(norm) or norm<=1e-12 else x/norm


def centers_for_k2(y, rng, *, init, edges=None):
    """Select *only* the initial two unit centers; never mutate EVA Y.

    Returns (centers, provenance). Randomness is supplied exclusively
    by the caller's restart-specific NumPy Generator.
    """
    x=np.asarray(y,dtype=np.float64)
    if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError("Invalid EVA vectors")
    n,d=x.shape
    if init not in INIT2:
        raise ValueError(f"Unsupported k=2 initializer {init}")
    if init in ("paired","independent"):
        # EXACT legacy behavior: draw a (2, p) normal matrix before
        # replacing row 1 by minus row 0 for K-MeansNM.
        centers=rng.standard_normal((2,d))
        if init=="paired":
            centers[1]=-centers[0]
        centers/=np.linalg.norm(centers,axis=1)[:,None]
        return centers,{"screened_hard_cuts":0}
    if init=="sample-antipodal":
        i=int(rng.integers(n))
        return np.stack((x[i],-x[i])),{
            "sample_vertex":i,"screened_hard_cuts":0,
        }
    if init=="sample-pair":
        indices=rng.choice(n,size=2,replace=False).astype(int)
        return x[indices].copy(),{
            "sample_vertices":indices.tolist(),"screened_hard_cuts":0,
        }
    if init=="farthest-pair":
        i=int(rng.integers(n))
        # On the unit sphere, the furthest Euclidean neighbor is
        # the one with MINIMUM dot product.
        distances=x@x[i]
        distances[i]=np.inf
        j=int(np.argmin(distances))
        return np.stack((x[i],x[j])),{
            "sample_vertices":[i,j],"screened_hard_cuts":0,
        }
    if init=="spherical-pp":
        i=int(rng.integers(n))
        # Standard k-means++ D^2 on unit vectors: ||Yi-Yj||^2
        # = 2(1-cos(angle)). This is geometry-only.
        weights=np.maximum(0.,2.*(1.-x@x[i]))
        weights[i]=0.
        total=float(weights.sum())
        if total<=0:
            j=(i+1)%n
        else:
            j=int(rng.choice(n,p=weights/total))
        return np.stack((x[i],x[j])),{
            "sample_vertices":[i,j],"screened_hard_cuts":0,
        }
    if init=="mean-perturb":
        mean=_unit(x.mean(axis=0))
        if mean is None:
            mean=_unit(rng.standard_normal(d))
        v=rng.standard_normal(d)
        v=v-mean*np.dot(v,mean)
        tangent=_unit(v)
        if tangent is None:
            raise ValueError("Could not find mean-orthogonal tangent")
        # Different but nearby spherical centers. Both are biased
        # toward the mean, unlike antipodal K-MeansNM.
        c0=_unit(mean-.35*tangent)
        c1=_unit(mean+.35*tangent)
        return np.stack((c0,c1)),{
            "perturbation":.35,"screened_hard_cuts":0,
        }
    if init=="pca-antipodal":
        # Principal direction computed from the EVA factor only,
        # not from graph cut labels. Stochastic jitter gives
        # distinct starts and avoids single repeated PCA split.
        mean=x.mean(axis=0)
        centered=x-mean
        gram=centered.T@centered
        eigenvalues,eigenvectors=np.linalg.eigh(gram)
        direction=eigenvectors[:,-1]
        # eigensolver's arbitrary global sign is irrelevant to Max-Cut.
        jitter=.25*rng.standard_normal(d)
        direction=_unit(direction+jitter)
        return np.stack((direction,-direction)),{
            "leading_pca_eigenvalue":float(eigenvalues[-1]),
            "jitter":.25,"screened_hard_cuts":0,
        }
    if init=="gw-screen16":
        if edges is None:
            raise ValueError("Cut-aware initialization requires original signed edges")
        best_score=-np.inf
        best_direction=None
        for _ in range(16):
            direction=_unit(rng.standard_normal(d))
            labels=(x@direction>0).astype(np.int8)
            score=float(cut_weight(edges,labels))
            if score>best_score:
                best_score=score
                best_direction=direction
        return np.stack((best_direction,-best_direction)),{
            "screened_hard_cuts":16,
            "best_screened_cut":float(best_score),
        }
    raise AssertionError("Unreachable initializer")
