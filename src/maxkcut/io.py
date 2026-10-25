"""Load and validate canonical G-set text and legacy MATLAB weighted graphs."""
from __future__ import annotations
from pathlib import Path
from inspect import signature
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

def _load_gset_txt(path: Path) -> sparse.csr_matrix:
    """Read canonical G-set text: 'n m' then m one-based 'i j weight' rows.

    The graph is undirected and each edge must appear exactly once. A genuine
    G-set graph may have negative weights, which are preserved unmodified.
    The header's n is authoritative, including isolated highest-numbered
    vertices that may not occur in the edge list.
    """
    with Path(path).open("r", encoding="utf8") as stream:
        header = stream.readline().split()
    if len(header) != 2:
        raise ValueError(f"{path}: expected G-set header 'n m'")
    try:
        n, m = map(int, header)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid G-set vertex/edge counts") from exc
    if n < 2 or m < 1:
        raise ValueError(f"{path}: G-set header requires n >= 2 and m >= 1")

    try:
        rows = np.loadtxt(path, dtype=float, skiprows=1, ndmin=2)
    except (ValueError, OSError) as exc:
        raise ValueError(f"{path}: invalid G-set edge list: {exc}") from exc
    if rows.shape != (m, 3) or not np.isfinite(rows).all():
        raise ValueError(
            f"{path}: expected exactly {m} finite rows of 3 columns; "
            f"found {rows.shape}"
        )
    identifiers = rows[:, :2]
    if not np.array_equal(identifiers, np.floor(identifiers)):
        raise ValueError(f"{path}: vertex identifiers must be integers")
    if np.any(identifiers < 1) or np.any(identifiers > n):
        raise ValueError(f"{path}: vertex identifiers must be in 1..{n}")
    ij = identifiers.astype(np.int64) - 1
    if np.any(ij[:, 0] == ij[:, 1]):
        raise ValueError(f"{path}: self-loops are not supported")
    left = np.minimum(ij[:, 0], ij[:, 1])
    right = np.maximum(ij[:, 0], ij[:, 1])
    keys = left * n + right
    if len(np.unique(keys)) != m:
        raise ValueError(f"{path}: duplicate or reversed duplicate edge")
    weighted = sparse.coo_matrix(
        (rows[:, 2], (left, right)), shape=(n, n), dtype=float,
    ).tocsr()
    result = weighted + weighted.T
    result.eliminate_zeros()
    return result.tocsr()


def load_graph(path: Path) -> sparse.csr_matrix:
    """Load a canonical G-set text graph or MATLAB v5 adjacency/edge list.

    Refuse ambiguous candidate matrices, nonzero diagonal and asymmetric weights.
    A triangular adjacency is symmetrized; a symmetric adjacency is not doubled.
    """
    path = Path(path)
    if path.suffix.lower() == ".txt":
        return _load_gset_txt(path)
    if path.suffix.lower() != ".mat":
        raise ValueError(f"Unsupported graph format {path.suffix}; use .txt or .mat")
    # Since SciPy 1.18, loadmat warns when the default sparse return type is
    # left unspecified. Sparse arrays are supported below because we normalize
    # to csr_matrix. Older supported SciPy releases lack this parameter.
    load_options = {"simplify_cells": True}
    if "spmatrix" in signature(loadmat).parameters:
        load_options["spmatrix"] = False
    data = loadmat(path, **load_options)
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
    """Return three arrays (i, j, weight), one per undirected edge.

    Only upper-triangular entries are counted to avoid double counting.
    """
    upper = sparse.triu(a, 1).tocoo()
    return upper.row, upper.col, upper.data
