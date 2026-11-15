"""Independent analytical-primal/dual checks and EVA trace (NumPy only).

Run from monograph root:
    python code/check_examples.py

Writes experiments/experiment001/results.json and figures/eva_wheel6_trace.png.
No large original maxkcut experiment is executed or overwritten.
"""
from __future__ import annotations
import itertools
import json
from math import acos, cos, pi, sqrt
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'experiments' / 'experiment001'


def cycle5() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    n = 5
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i + 1) % n] = W[(i + 1) % n, i] = 1.0
    L = np.diag(W.sum(axis=1)) - W
    theta = np.arange(n) * 4 * pi / 5
    Y = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    c = cos(pi / 5)
    nu = np.full(n, (1 + c) / 2)
    X, Z = Y @ Y.T, np.diag(nu) - L / 4
    info = report(W, Y, nu, 'cycle5')
    expected = 2.5 * (1 + c)
    assert abs(info['primal'] - expected) < 1e-12
    assert abs(info['dual'] - expected) < 1e-12
    assert info['discrete_opt'] == 4.0
    return W, Y, X, Z, info


def wheel6(q: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    n = 6
    W = np.zeros((n, n))
    for i in range(5):
        W[i, (i + 1) % 5] = W[(i + 1) % 5, i] = 1.0
        W[i, 5] = W[5, i] = float(q)
    L = np.diag(W.sum(axis=1)) - W
    c = cos(pi / 5)
    assert 0 < q <= 2 * (1 + c)
    a = q / (2 * (1 + c))
    theta = np.arange(5) * 4 * pi / 5
    Y = np.zeros((n, 3))
    Y[:5, 0] = sqrt(1 - a * a) * np.cos(theta)
    Y[:5, 1] = sqrt(1 - a * a) * np.sin(theta)
    Y[:5, 2] = a
    Y[5, 2] = -1
    alpha = (2 + q + 2 * c) / 4
    beta = 5 * q * (1 + a) / 4
    nu = np.array([alpha] * 5 + [beta])
    X, Z = Y @ Y.T, np.diag(nu) - L / 4
    info = report(W, Y, nu, f'wheel6_q{q:g}')
    expected = 2.5 * (1 + c + q) + 5 * q*q / (8 * (1 + c))
    assert abs(info['primal'] - expected) < 1e-12
    assert abs(info['dual'] - expected) < 1e-12
    return W, Y, X, Z, info


def continuous_score(W: np.ndarray, Y: np.ndarray) -> float:
    I, J = np.triu_indices(len(W), 1)
    return float(0.5 * np.dot(W[I, J], 1 - np.sum(Y[I] * Y[J], axis=1)))


def discrete_opt(W: np.ndarray) -> tuple[float, list[int]]:
    n = len(W)
    best = -float('inf')
    best_spins = None
    for signs in itertools.product((-1, 1), repeat=n):
        s = np.array(signs, dtype=float)
        score = continuous_score(W, s[:, None])
        if score > best:
            best = score
            best_spins = list(signs)
    assert best_spins is not None
    return float(best), best_spins


def report(W: np.ndarray, Y: np.ndarray, nu: np.ndarray, name: str) -> dict:
    L = np.diag(W.sum(axis=1)) - W
    X = Y @ Y.T
    Z = np.diag(nu) - L / 4
    optimum, spin = discrete_opt(W)
    info = {
        'example': name,
        'n': len(W),
        'm': int(np.count_nonzero(np.triu(W, 1))),
        'primal': continuous_score(W, Y),
        'dual': float(nu.sum()),
        'discrete_opt': optimum,
        'optimal_spins': spin,
        'unit_residual': float(np.max(np.abs(np.linalg.norm(Y, axis=1) - 1))),
        'primal_min_eig': float(np.linalg.eigvalsh(X)[0]),
        'dual_min_eig': float(np.linalg.eigvalsh(Z)[0]),
        'complementarity_fro': float(np.linalg.norm(Z @ Y, ord='fro')),
        'primal_dual_gap': float(nu.sum() - continuous_score(W, Y)),
        'primal_rank': int(np.linalg.matrix_rank(X, tol=1e-9)),
    }
    assert info['unit_residual'] < 1e-12, info
    assert info['primal_min_eig'] > -1e-12, info
    assert info['dual_min_eig'] > -1e-12, info
    assert info['complementarity_fro'] < 1e-11, info
    assert abs(info['primal_dual_gap']) < 1e-11, info
    return info


def eva(W: np.ndarray, p: int, seed: int = 2026,
        tolerance: float = 1e-6, max_epochs: int = 10000) -> tuple[np.ndarray, list[dict]]:
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    Y = rng.random((n, p), dtype=np.float64)
    Y /= np.linalg.norm(Y, axis=1)[:, None]
    trace: list[dict] = []
    previous = continuous_score(W, Y)
    for epoch in range(1, max_epochs + 1):
        max_angle = 0.0
        for i in range(n):
            old = Y[i].copy()
            h = W[i] @ Y
            norm = float(np.linalg.norm(h))
            if norm < 1e-15:
                fresh = rng.random(p)
                fresh /= np.linalg.norm(fresh)
            else:
                fresh = -h / norm
            theta = float(np.arccos(np.clip(old @ fresh, -1, 1)))
            max_angle = max(max_angle, theta)
            Y[i] = fresh
        current = continuous_score(W, Y)
        assert current >= previous - 1e-10, (epoch, current, previous)
        trace.append({'epoch': epoch, 'objective': current, 'max_angle_rad': max_angle})
        previous = current
        if max_angle < tolerance:
            break
    return Y, trace


def draw_plot(trace: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.plot([r['epoch'] for r in trace], [r['objective'] for r in trace], marker='o', ms=3)
    c = cos(pi/5)
    optimum = 2.5 * (2 + c) + 5/(8*(1+c))
    ax.axhline(optimum, linestyle='--', linewidth=1.1, label='Exact primal-dual optimum')
    ax.set(xlabel='Completed EVA epoch', ylabel='Continuous Max-Cut objective',
           title='Sequential EVA on the unit-weight wheel W6')
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    (ROOT / 'figures').mkdir(exist_ok=True)
    fig.savefig(ROOT / 'figures' / 'eva_wheel6_trace.png', dpi=170)
    fig.savefig(ROOT / 'figures' / 'eva_wheel6_trace.pdf')
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wc, Yc, Xc, Zc, cinfo = cycle5()
    wheel_results = [wheel6(q)[-1] for q in (0.5, 1, 2)]
    W, Ystar, Xstar, Zstar, winfo = wheel6(1)
    Yeva, trace = eva(W, p=4, seed=2026, tolerance=1e-6)
    result = {
        'description': 'Independent analytic primal-dual certificates and EVA trace; no original repository factor modified',
        'seed': 2026,
        'eva_rank': 4,
        'eva_tolerance': 1e-6,
        'cycle5': cinfo,
        'wheel6': winfo,
        'wheel6_parameter_sweep': wheel_results,
        'wheel6_eva': {
            'epochs': len(trace),
            'objective': continuous_score(W, Yeva),
            'gram_distance_fro': float(np.linalg.norm(Yeva @ Yeva.T - Xstar, 'fro')),
            'selected_trace': [r for r in trace if r['epoch'] in (1, 2, 3, 5, 10, 20)],
            'full_trace': trace,
        },
    }
    (OUT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    draw_plot(trace)
    print('Five-cycle SDP objective: ', cinfo['primal'], 'discrete:', cinfo['discrete_opt'])
    print('Wheel SDP objective:      ', winfo['primal'], 'discrete:', winfo['discrete_opt'])
    print('EVA objective:            ', result['wheel6_eva']['objective'])
    print('EVA epochs:               ', len(trace))
    print('Gram discrepancy:         ', result['wheel6_eva']['gram_distance_fro'])
    for r in result['wheel6_eva']['selected_trace']:
        print(f"epoch {r['epoch']:3d}: {r['objective']:.12f}  angle={r['max_angle_rad']:.12f}")
    print('Saved:', OUT / 'results.json')

if __name__ == '__main__':
    main()
