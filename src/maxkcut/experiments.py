"""Repeatable G-set benchmark orchestration and JSONL output."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from .io import load_graph, edge_arrays
from .objective import cut_weight
from .rounding import factorize, round_fj, round_spherical_kmeans
from .sdp import solve_sdp

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
