"""Reproduce G-set SDP rounding experiments; no solver work at import time."""
from __future__ import annotations

import argparse
import json
import time
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


def cut_weight(edges, labels):
    i, j, weights = edges
    return float(np.sum(weights[labels[i] != labels[j]]))


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


def solve_sdp(a, k, solver, eps, max_iters, verbose=False):
    """Use a *real convex SDP*, not a low-rank nonconvex proxy."""
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError('Install CVXPY first: python -m pip install -r requirements.txt') from exc
    n = a.shape[0]
    degrees = np.asarray(a.sum(axis=1)).ravel()
    laplacian = sparse.diags(degrees) - a
    x = cp.Variable((n, n), PSD=True)
    constraints = [cp.diag(x) == 1]
    if k > 2:
        constraints.append(x >= -1 / (k - 1))
    objective = cp.Maximize((k - 1) / (2 * k) * cp.sum(cp.multiply(laplacian, x)))
    problem = cp.Problem(objective, constraints)
    kwargs = {'verbose': verbose}
    if solver == 'SCS':
        kwargs.update(eps=eps, max_iters=max_iters)
    start = time.monotonic()
    value = problem.solve(solver=solver, **kwargs)
    seconds = time.monotonic() - start
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or x.value is None:
        raise RuntimeError(f'SDP solver status: {problem.status}')
    matrix = np.asarray(x.value, dtype=float)
    residuals = {'min_eigenvalue': float(np.linalg.eigvalsh((matrix+matrix.T)/2)[0]),
                 'max_diagonal_error': float(np.max(np.abs(np.diag(matrix)-1))),
                 'max_kway_violation': float(max(0, -1/(k-1) - np.min(matrix)))}
    return matrix, {'sdp_objective_primal': float(value), 'solver_status': problem.status,
                    'solver': solver, 'sdp_seconds': seconds, **residuals}


def benchmark(paths, k_values, output, *, solver='SCS', eps=1e-3,
              max_iters=3000, restarts=20, seed=2026, max_n=1000, verbose=False):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('a', encoding='utf8') as file:
        for path in paths:
            a = load_graph(path)
            n = a.shape[0]
            if n > max_n:
                raise ValueError(f'{path.name}: n={n} > --max-n={max_n}; raise limit explicitly')
            edges = edge_arrays(a)
            for k in k_values:
                if not 2 <= k <= n:
                    raise ValueError(f'Require 2 <= k <= n; got k={k}, n={n}')
                x, solver_info = solve_sdp(a, k, solver, eps, max_iters, verbose=verbose)
                vectors, min_before = factorize(x)
                methods = ('FJ', 'K-Means2N', 'K-MeansNM') if k == 2 else ('FJ', 'spherical_kmeans')
                for method in methods:
                    best = None
                    for rep in range(restarts):
                        rng = np.random.default_rng(np.random.SeedSequence([seed, int(path.stem[1:]), k, rep, 0 if method in ('FJ', 'K-MeansNM') else 1]))
                        started = time.monotonic()
                        if method == 'FJ':
                            labels = round_fj(vectors, k, rng)
                            iterations = 1
                        else:
                            labels, iterations = round_spherical_kmeans(
                                vectors, k, rng,
                                antipodal=(method == 'K-MeansNM'),
                                edges=edges if method == 'K-MeansNM' else None,
                                keep_best=(method == 'K-MeansNM'))
                        elapsed = time.monotonic() - started
                        score = cut_weight(edges, labels)
                        entry = {'graph': path.stem, 'n': n, 'edges': int(len(edges[0])), 'k': k,
                                 'method': method, 'restart': rep, 'cut_weight': score,
                                 'num_nonempty_clusters': int(len(np.unique(labels))),
                                 'rounding_seconds': elapsed, 'rounding_iterations': iterations,
                                 'seed': seed, 'factor_min_eigenvalue_before_clipping': min_before,
                                 **solver_info}
                        file.write(json.dumps(entry, allow_nan=False) + '\n')
                        file.flush()
                        if best is None or score > best['cut_weight']:
                            best = entry
                    print(f"{path.stem} k={k} {method}: best={best['cut_weight']} / {restarts} restarts", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphs', nargs='+', default=['G11'], help='Graph names, e.g. G11 G1 G22; ALL selects every file')
    parser.add_argument('--gset-dir', type=Path, default=Path('G-Set'))
    parser.add_argument('--k', type=int, nargs='+', default=[2])
    parser.add_argument('--output', type=Path, default=Path('reports/gset/runs.jsonl'))
    parser.add_argument('--solver', choices=['SCS', 'CLARABEL', 'MOSEK'], default='SCS')
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--max-iters', type=int, default=3000)
    parser.add_argument('--restarts', type=int, default=20)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-n', type=int, default=1000)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--inspect-only', action='store_true', help='Validate and report .mat graph sizes without CVXPY')
    args = parser.parse_args()
    names = sorted(args.gset_dir.glob('G*.mat')) if args.graphs == ['ALL'] else [args.gset_dir / (s if s.endswith('.mat') else s + '.mat') for s in args.graphs]
    for name in names:
        if not name.is_file():
            parser.error(f'No graph file: {name}')
    if args.inspect_only:
        for name in names:
            adjacency = load_graph(name)
            _, _, weights = edge_arrays(adjacency)
            print(json.dumps({'graph': name.stem, 'n': adjacency.shape[0],
                              'edges': int(len(weights)), 'weight_min': float(weights.min()),
                              'weight_max': float(weights.max())}))
        return
    if args.restarts < 1:
        parser.error('--restarts must be positive')
    benchmark(names, args.k, args.output, solver=args.solver, eps=args.eps,
              max_iters=args.max_iters, restarts=args.restarts, seed=args.seed,
              max_n=args.max_n, verbose=args.verbose)


if __name__ == '__main__':
    main()
