"""Original graph cut objective (not a surrogate clustering objective)."""
from __future__ import annotations
import numpy as np

def cut_weight(edges, labels):
    """Return the exact weighted cut of the provided graph partition.

    Parameters
    ----------
    edges : tuple
        (source_indices, target_indices, weights) from :func:`edge_arrays`.
    labels : array_like
        A class label for each graph vertex.

    Returns
    -------
    float
        The sum of weights for edges crossing between different classes.
        Negative weights contribute with their sign; this is *not* k-means
        inertia or another surrogate.
    """
    i, j, weights = edges
    return float(np.sum(weights[labels[i] != labels[j]]))
