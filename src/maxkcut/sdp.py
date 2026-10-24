"""Frieze–Jerrum / Goemans–Williamson semidefinite relaxation via CVXPY."""
from __future__ import annotations
import time
import numpy as np
from scipy import sparse

def solve_sdp(a, k, solver, eps, max_iters, verbose=False):
    """Solve the Goemans–Williamson / Frieze–Jerrum convex SDP.

    Parameters
    ----------
    a : scipy.sparse matrix
        Symmetric weighted adjacency, without self-loops.
    k : int
        Number of classes. For k>2 the Gram lower bound is -1/(k-1).
    solver : str
        CVXPY-compatible PSD cone solver, normally SCS.
    eps, max_iters : float, int
        SCS accuracy and iteration controls.
    verbose : bool
        Forward numerical solver logs.

    Returns
    -------
    tuple
        Numerical Gram matrix X and diagnostics dict. The reported *primal*
        SDP objective is not automatically a certified upper bound.
    """
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
