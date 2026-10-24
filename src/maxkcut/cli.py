"""Command-line interface for inspecting graph inputs and launching experiments."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .io import load_graph, edge_arrays
from .experiments import benchmark

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
