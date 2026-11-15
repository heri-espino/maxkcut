"""Optional small SDP solve using CVXPY, plus EVA + both factor representations.

Run: pip install numpy cvxpy scs
     python code/solve_sdp_compare.py

Uses the same continuous objective but two different computational routes.
No claims of rigorous certificates based solely on floating-point solver status.
"""
import numpy as np
from check_examples import cycle5, wheel6, eva, continuous_score

try:
    import cvxpy as cp
except ImportError as e:
    raise SystemExit('Missing optional CVXPY. Install with: pip install cvxpy scs') from e


def spectral_factor(X, negative_eigenvalue_tol=1e-5):
    S = (X + X.T) / 2
    eigenvalues, Q = np.linalg.eigh(S)
    if eigenvalues.min() < -negative_eigenvalue_tol:
        raise ValueError(f'Numerical SDP matrix not PSD: {eigenvalues.min()}')
    V = Q * np.sqrt(np.maximum(eigenvalues, 0))[None, :]
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V, eigenvalues


def solve(W):
    n = len(W)
    L = np.diag(W.sum(axis=1)) - W
    X = cp.Variable((n, n), symmetric=True)
    problem = cp.Problem(cp.Maximize(cp.trace(L @ X) / 4), [cp.diag(X) == 1, X >> 0])
    primal = problem.solve(solver=cp.SCS, eps=1e-8, max_iters=100000)
    if X.value is None:
        raise RuntimeError(f'CVXPY solve failed: {problem.status}')
    V, eig = spectral_factor(X.value)
    return V, {
        'status': problem.status,
        'cvxpy_primal_reported': primal,
        'postprocessed_factor_objective': continuous_score(W, V),
        'sdp_min_eigenvalue_before_repair': float(eig.min()),
        'diagonal_residual_before_repair': float(np.max(np.abs(np.diag(X.value) - 1))),
    }


def main():
    for name, constructor in [('C5', cycle5), ('W6', wheel6)]:
        W, Yanalytic, Xanalytic, Zanalytic, info = constructor()
        V, diag = solve(W)
        Y, trace = eva(W, p=min(len(W), 4), seed=2026)
        print('\n', name, '\n', diag)
        print('Analytical SDP optimum:', info['primal'])
        print('EVA factor value:', continuous_score(W, Y), 'epochs:', len(trace))
        print('CVX Gram error to a chosen analytic optimal Gram:', np.linalg.norm(V @ V.T-Xanalytic))
        print('EVA Gram error to a chosen analytic optimal Gram:', np.linalg.norm(Y @ Y.T-Xanalytic))

if __name__ == '__main__':
    main()
