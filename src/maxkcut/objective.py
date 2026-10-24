"""Original graph cut objective (not a surrogate clustering objective)."""
from __future__ import annotations
import numpy as np

def cut_weight(edges, labels):
    i, j, weights = edges
    return float(np.sum(weights[labels[i] != labels[j]]))
