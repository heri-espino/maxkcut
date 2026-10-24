"""PSD factorization and randomized or spherical k-means rounding."""
from __future__ import annotations
import numpy as np
from .objective import cut_weight

def factorize(x):
    """PSD eigendecomposition with documented, small-negative-eigenvalue clipping."""
    x = (x + x.T) / 2
    eigenvalues, q = np.linalg.eigh(x)
    min_eigenvalue = float(eigenvalues[0])
    scale = max(1., float(np.max(np.abs(eigenvalues))))
    if min_eigenvalue < -1e-5 * scale:
        raise ValueError(f'SDP output is substantially non-PSD: {min_eigenvalue}')
    clipped = np.maximum(eigenvalues, 0)
    vectors = q * np.sqrt(clipped)[None, :]
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms < 1e-10):
        raise ValueError('Zero-length vector after PSD repair')
    vectors = vectors / norms[:, None]
    return vectors, min_eigenvalue

def round_fj(vectors, k, rng):
    # For k=2 use the thesis's GW random hyperplane, not two Gaussian directions.
    if k == 2:
        direction = rng.standard_normal(vectors.shape[1])
        return (vectors @ direction >= 0).astype(int)
    centers = rng.standard_normal((k, vectors.shape[1]))
    return np.argmax(vectors @ centers.T, axis=1)

def round_spherical_kmeans(vectors, k, rng, max_iter=100, *, antipodal=False,
                           edges=None, keep_best=False):
    """Spherical k-means; antipodal initialization reproduces K-MeansNM at k=2.

    If keep_best, retain the highest graph cut over every clustering iteration,
    including the initial assignment. This is key for GW baseline retention.
    """
    if antipodal and k != 2:
        raise ValueError('Antipodal initialization applies only to k=2')
    if keep_best and edges is None:
        raise ValueError('Tracking the best cut requires the original graph edges')
    centers = rng.standard_normal((k, vectors.shape[1]))
    if antipodal:
        centers[1] = -centers[0]
    centers /= np.linalg.norm(centers, axis=1)[:, None]
    previous = None
    best_labels, best_score = None, float('-inf')
    for iteration in range(1, max_iter + 1):
        labels = np.argmax(vectors @ centers.T, axis=1)
        if keep_best:
            score = cut_weight(edges, labels)
            if score > best_score:
                best_labels, best_score = labels.copy(), score
        if previous is not None and np.array_equal(labels, previous):
            return (best_labels if keep_best else labels), iteration
        previous = labels.copy()
        for c in range(k):
            members = vectors[labels == c]
            if not len(members):
                centers[c] = vectors[rng.integers(len(vectors))]
            else:
                center = np.sum(members, axis=0)
                norm = np.linalg.norm(center)
                centers[c] = center / norm if norm > 1e-12 else vectors[rng.integers(len(vectors))]
    return (best_labels if keep_best else labels), max_iter
