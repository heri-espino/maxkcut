"""Load and validate undirected weighted graphs from MATLAB v5 files."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.io import loadmat

def _matrix_candidates(obj, prefix=''):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not key.startswith('__'):
                yield from _matrix_candidates(value, f'{prefix}.{key}' if prefix else key)
    elif sparse.issparse(obj) or (isinstance(obj, np.ndarray) and obj.ndim == 2):
        yield prefix, obj

def load_graph(path: Path) -> sparse.csr_matrix:
    """Load a MATLAB v5 sparse/dense adjacency or 3-column edge list.

    Refuse ambiguous candidate matrices, nonzero diagonal and asymmetric weights.
    A triangular adjacency is symmetrized; a symmetric adjacency is not doubled.
    """
    data = loadmat(path, simplify_cells=True)
    candidates = [(key, value) for key, value in _matrix_candidates(data)
                  if value.shape[0] > 1 and value.shape[1] > 1]
    square = [(key, value) for key, value in candidates if value.shape[0] == value.shape[1]]
    if square:
        names = ('A', 'W', 'adj', 'adjacency', 'matrix', 'G')
        preferred = [(key, value) for key, value in square
                     if key.split('.')[-1] in names]
        options = preferred or square
        if len(options) != 1:
            raise ValueError(f'Ambiguous square matrices in {path}: {[key for key, _ in options]}')
        a = sparse.csr_matrix(options[0][1], dtype=float)
    else:
        edges = [(key, value) for key, value in candidates if value.shape[1] == 3]
        if len(edges) != 1:
            raise ValueError(f'Expected one adjacency matrix or one m-by-3 edge list; found {[(k,v.shape) for k,v in candidates]}')
        x = np.asarray(edges[0][1], dtype=float)
        ij = x[:, :2].astype(int)
        if not np.array_equal(ij, x[:, :2]) or np.any(ij < 0):
            raise ValueError('Invalid node identifiers in edge list')
        offset = 0 if np.any(ij == 0) else 1  # G-set text convention is 1-based
        ij -= offset
        n = int(ij.max()) + 1
        if np.any(ij[:, 0] == ij[:, 1]):
            raise ValueError('Self-loops are not supported')
        upper = sparse.coo_matrix((x[:, 2], (ij[:, 0], ij[:, 1])), shape=(n, n)).tocsr()
        # Edge lists must contain each undirected edge only once.
        if upper.nnz != len(x) or (upper.multiply(upper.T)).nnz:
            raise ValueError('Repeated or duplicated reverse-direction edges')
        a = upper + upper.T
    a.sum_duplicates()
    if not np.isfinite(a.data).all() or np.any(a.diagonal() != 0):
        raise ValueError('Nonfinite edge weights or nonzero diagonal')
    diff = a - a.T
    if diff.nnz:
        if (sparse.triu(a, 1).nnz and sparse.tril(a, -1).nnz):
            raise ValueError('Conflicting/asymmetric edge weights')
        a = a + a.T
    a.eliminate_zeros()
    return a.tocsr()

def edge_arrays(a: sparse.csr_matrix):
    upper = sparse.triu(a, 1).tocoo()
    return upper.row, upper.col, upper.data
