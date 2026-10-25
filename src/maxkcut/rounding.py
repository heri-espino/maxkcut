"""PSD factorization and randomized or spherical k-means rounding."""
from __future__ import annotations
import numpy as np
from .objective import cut_weight

def factorize(x, *, relative_negative_tolerance=1e-3,
              return_diagnostics=False):
    """Factor a nearly-PSD numerical Gram matrix into unit row vectors.

    SCS and other first-order SDP solvers often return *approximately* PSD
    matrices. Small negative eigenvalues are clipped to zero (projection onto
    the PSD cone), and the resulting vectors are normalized to unit length.
    This is a feasible Gram representation for k=2, not an assertion that the
    raw SDP iterate was feasible or that its primal objective is an upper bound.

    A *large* negative eigenvalue remains an error. The acceptance threshold
    uses |lambda_min| / max(1, lambda_max), so it is invariant under a change
    in the number of nodes (for a unit-diagonal matrix lambda_max >= 1).

    For k>2 the PSD projection and renormalization need not preserve all the
    Frieze-Jerrum pairwise lower bounds. Returned vectors are useful for
    rounding but must not be presented as a certified feasible SDP solution.

    Parameters
    ----------
    x : array_like
        Approximately symmetric SDP Gram matrix.
    relative_negative_tolerance : float, optional
        Largest allowed relative negative eigenvalue (default 1e-3). Set to
        zero for strict numerical PSD checking.
    return_diagnostics : bool, optional
        Return a third dictionary containing the magnitude of repairs.

    Returns
    -------
    vectors, min_eigenvalue : ndarray, float
        Unit row vectors and smallest eigenvalue *before* clipping.
    diagnostics : dict, optional
        Returned only if return_diagnostics=True. Records the number and
        magnitude of negative eigenvalues and whether a repair was made.
    """
    import warnings

    matrix = np.asarray(x, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or not np.isfinite(matrix).all()):
        raise ValueError("Gram matrix must be finite and square")
    if not np.isfinite(relative_negative_tolerance) or relative_negative_tolerance < 0:
        raise ValueError("relative_negative_tolerance must be nonnegative and finite")

    sym = (matrix + matrix.T) / 2
    eigenvalues, q = np.linalg.eigh(sym)
    min_eigenvalue = float(eigenvalues[0])
    scale = max(1.0, float(eigenvalues[-1]))
    relative_negative = max(0.0, -min_eigenvalue / scale)
    if relative_negative > relative_negative_tolerance:
        raise ValueError(
            "SDP output is substantially non-PSD: "
            f"lambda_min={min_eigenvalue:.9g}, "
            f"relative_violation={relative_negative:.9g}, "
            f"allowed={relative_negative_tolerance:.9g}. "
            "Try --eps 1e-5 --max-iters 20000 --no-sdp-cache."
        )

    # Numerical negativity at the scale of floating point roundoff is not
    # treated as a noteworthy solver defect, but is still clipped.
    count_negative = int(np.count_nonzero(eigenvalues < 0))
    notable = relative_negative > 1e-10
    if notable:
        warnings.warn(
            "Numerically indefinite SDP matrix: clipped negative "
            f"eigenvalues (lambda_min={min_eigenvalue:.9g}, "
            f"relative={relative_negative:.3g}). "
            "The reported primal SDP objective is NOT a certified upper bound. "
            "Consider a tighter solver tolerance.",
            RuntimeWarning,
            stacklevel=2,
        )

    clipped = np.maximum(eigenvalues, 0)
    vectors = q * np.sqrt(clipped)[None, :]
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms < 1e-10):
        raise ValueError("Zero-length SDP vector after PSD projection")
    vectors = vectors / norms[:, None]
    diagnostics = {
        "psd_projection_applied": bool(notable),
        "psd_negative_eigenvalue_count": count_negative,
        "psd_negative_eigenvalue_sum": float(-np.sum(eigenvalues[eigenvalues < 0])),
        "psd_relative_negative_eigenvalue": float(relative_negative),
        "psd_original_max_eigenvalue": float(eigenvalues[-1]),
        "psd_original_max_diagonal_error": float(
            np.max(np.abs(np.diag(matrix) - 1))
        ),
        "psd_repair_tolerance": float(relative_negative_tolerance),
    }
    if return_diagnostics:
        return vectors, min_eigenvalue, diagnostics
    return vectors, min_eigenvalue

def round_fj(vectors, k, rng):
    """Round SDP vectors with GW for k=2 or Frieze–Jerrum for k>2.

    Parameters
    ----------
    vectors : ndarray
        Rows correspond to graph vertices, columns to Euclidean coordinates.
    k : int
        Number of allowable classes (at least 2).
    rng : numpy.random.Generator
        Explicit RNG; for k>2 Gaussian directions are *not* normalized.

    Returns
    -------
    ndarray
        Integer graph labels, with values between 0 and k-1.
    """
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
