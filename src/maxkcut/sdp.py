"""Frieze–Jerrum / Goemans–Williamson semidefinite relaxation via CVXPY."""
from __future__ import annotations
import time
import numpy as np
from scipy import sparse

def solve_sdp(a, k, solver, eps, max_iters, verbose=False, *,
              scs_indirect=False, compute_eigenvalues=True):
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
    scs_indirect : bool
        Use SCS's sparse iterative KKT solver. Not necessarily faster for SDPs;
        benchmark it explicitly against the direct solver.
    compute_eigenvalues : bool
        If False, skip the expensive PSD eigenvalue diagnostic. In a benchmark
        that immediately calls `factorize`, reuse that eigendecomposition instead.

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
    if k < 2:
        raise ValueError("k must be at least 2.")
    if scs_indirect and solver != "SCS":
        raise ValueError("scs_indirect is only supported with SCS.")
    installed = cp.installed_solvers()
    if solver not in installed:
        raise ValueError(f"Solver {solver} is not installed. Installed: {installed}")
    n = a.shape[0]
    degrees = np.asarray(a.sum(axis=1)).ravel()
    laplacian = sparse.diags(degrees, dtype=float) - a
    x = cp.Variable((n, n), PSD=True)
    constraints = [cp.diag(x) == 1]
    if k > 2:
        constraints.append(x >= -1 / (k - 1))
    objective = cp.Maximize((k - 1) / (2 * k) * cp.sum(cp.multiply(laplacian, x)))
    problem = cp.Problem(objective, constraints)
    kwargs = {'verbose': verbose}
    if solver == 'SCS':
        kwargs.update(eps=eps, max_iters=max_iters, use_indirect=bool(scs_indirect))
    start = time.monotonic()
    value = problem.solve(solver=solver, **kwargs)
    seconds = time.monotonic() - start
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or x.value is None:
        raise RuntimeError(f'SDP solver status: {problem.status}')
    matrix = np.asarray(x.value, dtype=float)
    # Factorization in the benchmarking path already computes the full spectrum.
    # Do not run a separate O(n^3) eigvalsh pass for that same solution.
    min_eigenvalue = (
        float(np.linalg.eigvalsh((matrix + matrix.T) / 2)[0])
        if compute_eigenvalues else None
    )
    residuals = {
        'min_eigenvalue': min_eigenvalue,
        'max_diagonal_error': float(np.max(np.abs(np.diag(matrix) - 1))),
        'max_kway_violation': float(max(0, -1 / (k - 1) - np.min(matrix))),
    }
    stats = problem.solver_stats
    return matrix, {
        'sdp_objective_primal': float(value),
        'solver_status': problem.status,
        'solver': solver,
        'sdp_seconds': seconds,
        'scs_indirect': bool(scs_indirect) if solver == 'SCS' else None,
        'solver_iterations': int(stats.num_iters) if stats.num_iters is not None else None,
        'solver_setup_seconds': float(stats.setup_time) if stats.setup_time is not None else None,
        'solver_internal_seconds': float(stats.solve_time) if stats.solve_time is not None else None,
        **residuals,
    }
