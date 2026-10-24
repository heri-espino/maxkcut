"""Repeatable G-set experiment orchestration with SDP reuse and stage timings."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .cache import cache_identity, cache_path, load_cached_sdp, save_cached_sdp
from .hardware import blas_limit
from .io import edge_arrays, load_graph
from .objective import cut_weight
from .rounding import factorize, round_fj, round_spherical_kmeans
from .sdp import solve_sdp


def benchmark(paths, k_values, output, *, solver='SCS', eps=1e-3,
              max_iters=3000, restarts=20, seed=2026, max_n=1000,
              verbose=False, scs_indirect=False, cache_dir=Path('reports/gset/sdp_cache'),
              use_cache=True, psd_repair_rel_tol=1e-3,
              cpu_threads=None, scs_cudss=False):
    """Benchmark rounders against the same SDP, reusing cached Gram matrices.

    Save raw, solver-produced Gram matrices keyed to input file checksum,
    solver version and all relevant settings. A cache hit skips the expensive
    conic solve, but still revalidates/factorizes the saved matrix. New rounding
    seeds and restarts do not invalidate a legitimate SDP cache entry.

    Results append to a JSONL file; choose a fresh output path to avoid
    confusing separate executions with independent samples.
    """
    if cpu_threads is not None and (isinstance(cpu_threads, bool) or
                                    not isinstance(cpu_threads, int) or cpu_threads < 1):
        raise ValueError('cpu_threads must be a positive integer')
    if scs_cudss and (solver != 'SCS' or scs_indirect):
        raise ValueError('scs_cudss needs SCS direct mode')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if restarts < 1 or max_n < 2:
        raise ValueError('restarts must be positive, max_n must be >= 2')
    if eps <= 0 or max_iters < 1:
        raise ValueError('eps and max_iters must be positive')
    if not np.isfinite(psd_repair_rel_tol) or psd_repair_rel_tol < 0:
        raise ValueError('psd_repair_rel_tol must be nonnegative and finite')

    with output.open('a', encoding='utf8') as file:
        for path in paths:
            path = Path(path)
            a = load_graph(path)
            n = a.shape[0]
            if n > max_n:
                raise ValueError(f'{path.name}: n={n} > --max-n={max_n}; raise limit explicitly')
            edges = edge_arrays(a)
            for k in k_values:
                if not 2 <= k <= n:
                    raise ValueError(f'Require 2 <= k <= n; got k={k}, n={n}')

                start = time.monotonic()
                loaded = None
                entry_path = None
                if use_cache:
                    identity = cache_identity(
                        path, k=k, solver=solver, eps=eps,
                        max_iters=max_iters, scs_indirect=scs_indirect,
                        cpu_threads=cpu_threads, scs_cudss=scs_cudss
                    )
                    entry_path = cache_path(path, cache_dir, identity)
                    loaded = load_cached_sdp(entry_path, identity, n=n)
                hit = loaded is not None
                if hit:
                    x, solver_info = loaded
                    solver_info = dict(solver_info)
                    original_solve_seconds = solver_info['sdp_seconds']
                    solver_info['sdp_seconds'] = 0.0
                else:
                    with blas_limit(cpu_threads):
                        extra_options = {'scs_cudss': True} if scs_cudss else {}
                        x, solver_info = solve_sdp(
                            a, k, solver, eps, max_iters, verbose=verbose,
                            scs_indirect=scs_indirect, compute_eigenvalues=False,
                            **extra_options,
                        )
                    solver_info = dict(solver_info)
                    original_solve_seconds = solver_info['sdp_seconds']
                prep_seconds = time.monotonic() - start

                factor_start = time.monotonic()
                with blas_limit(cpu_threads):
                    vectors, min_before, repair_info = factorize(
                        x,
                        relative_negative_tolerance=psd_repair_rel_tol,
                        return_diagnostics=True,
                    )
                factor_seconds = time.monotonic() - factor_start
                # Do not persist a new matrix until the numerical validity
                # threshold has been checked. Original raw X is cached, not the
                # postprocessed Gram matrix from the repaired vectors.
                if use_cache and not hit:
                    save_cached_sdp(entry_path, identity, x, solver_info)
                del x
                solver_info['min_eigenvalue'] = min_before
                solver_info.update(repair_info)
                solver_info['cpu_threads_requested'] = cpu_threads
                solver_info['scs_cudss_requested'] = bool(scs_cudss)
                solver_info['cache_hit'] = hit
                solver_info['cache_path'] = str(entry_path) if entry_path else None
                solver_info['sdp_original_solve_seconds'] = original_solve_seconds
                solver_info['sdp_prepare_seconds'] = prep_seconds
                solver_info['factorization_seconds'] = factor_seconds

                methods = (
                    ('FJ', 'K-Means2N', 'K-MeansNM') if k == 2
                    else ('FJ', 'spherical_kmeans')
                )
                for method in methods:
                    best = None
                    for rep in range(restarts):
                        rng = np.random.default_rng(
                            np.random.SeedSequence([
                                seed, int(path.stem[1:]), k, rep,
                                0 if method in ('FJ', 'K-MeansNM') else 1
                            ])
                        )
                        started = time.monotonic()
                        if method == 'FJ':
                            labels = round_fj(vectors, k, rng)
                            iterations = 1
                        else:
                            labels, iterations = round_spherical_kmeans(
                                vectors, k, rng,
                                antipodal=(method == 'K-MeansNM'),
                                edges=edges if method == 'K-MeansNM' else None,
                                keep_best=(method == 'K-MeansNM')
                            )
                        elapsed = time.monotonic() - started
                        score = cut_weight(edges, labels)
                        entry = {
                            'graph': path.stem,
                            'n': n,
                            'edges': int(len(edges[0])),
                            'k': k,
                            'method': method,
                            'restart': rep,
                            'cut_weight': score,
                            'num_nonempty_clusters': int(len(np.unique(labels))),
                            'rounding_seconds': elapsed,
                            'rounding_iterations': iterations,
                            'seed': seed,
                            'factor_min_eigenvalue_before_clipping': min_before,
                            **solver_info
                        }
                        file.write(json.dumps(entry, allow_nan=False) + '\n')
                        file.flush()
                        if best is None or score > best['cut_weight']:
                            best = entry
                    print(
                        f"{path.stem} k={k} {method}: best={best['cut_weight']} / "
                        f"{restarts} restarts (SDP cache hit={hit})",
                        flush=True
                    )
