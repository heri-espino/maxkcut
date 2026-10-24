"""Command-line interface for inspecting graph inputs and launching experiments."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .hardware import hardware_info
from .io import load_graph, edge_arrays
from .experiments import benchmark

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphs', nargs='+', default=['G11'], help='Graph names, e.g. G11 G1 G22; ALL selects every file')
    parser.add_argument('--gset-dir', type=Path, default=Path('G-Set'))
    parser.add_argument('--k', type=int, nargs='+', default=[2])
    parser.add_argument('--output', type=Path, default=Path('reports/gset/runs.jsonl'))
    parser.add_argument('--solver', choices=['SCS', 'CLARABEL', 'MOSEK', 'SDPA'], default='SCS')
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--max-iters', type=int, default=3000)
    parser.add_argument('--restarts', type=int, default=20)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-n', type=int, default=1000)
    parser.add_argument('--scs-indirect', action='store_true',
                        help='Use SCS iterative sparse KKT solver (benchmark against direct)')
    parser.add_argument('--cpu-threads', type=int,
                        help='Limit recognized BLAS/LAPACK CPU thread pools during the SDP and eigendecomposition; benchmark 4/8/16')
    parser.add_argument('--scs-cudss', action='store_true',
                        help='Use GPU cuDSS SCS linear backend, ONLY if SCS is custom built with cuDSS and CUDA is available')
    parser.add_argument('--hardware-info', action='store_true',
                        help='Print detected logical CPUs, BLAS pools, and NVIDIA driver visibility')
    parser.add_argument('--cache-dir', type=Path,
                        default=Path('reports/gset/sdp_cache'),
                        help='Directory for safely fingerprinted SDP matrix cache')
    parser.add_argument('--no-sdp-cache', action='store_true',
                        help='Re-solve the SDP instead of reading/writing cached matrices')
    parser.add_argument('--psd-repair-rel-tol', type=float, default=1e-3,
                        help='Maximum tolerable negative Gram eigenvalue relative to spectral scale (default: 1e-3)')
    parser.add_argument('--list-solvers', action='store_true',
                        help='Show locally installed CVXPY solvers and exit')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--inspect-only', action='store_true', help='Validate and report .mat graph sizes without CVXPY')
    args = parser.parse_args()
    if args.hardware_info:
        print(json.dumps(hardware_info(), indent=2))
        return
    if args.list_solvers:
        try:
            import cvxpy as cp
        except ImportError as exc:
            parser.error(f'CVXPY is not installed: {exc}')
        print(json.dumps({'installed_solvers': cp.installed_solvers()}))
        return
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    if args.scs_cudss and (args.solver != 'SCS' or args.scs_indirect):
        parser.error('--scs-cudss requires --solver SCS without --scs-indirect')
    if args.scs_indirect and args.solver != 'SCS':
        parser.error('--scs-indirect requires --solver SCS')
    if args.psd_repair_rel_tol < 0 or not __import__('math').isfinite(args.psd_repair_rel_tol):
        parser.error('--psd-repair-rel-tol must be nonnegative and finite')
    if args.eps <= 0 or args.max_iters < 1:
        parser.error('--eps and --max-iters must be positive')
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
              max_n=args.max_n, verbose=args.verbose,
              scs_indirect=args.scs_indirect, cache_dir=args.cache_dir,
              use_cache=not args.no_sdp_cache,
              psd_repair_rel_tol=args.psd_repair_rel_tol,
              cpu_threads=args.cpu_threads, scs_cudss=args.scs_cudss)
