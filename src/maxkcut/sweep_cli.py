"""First-pass G-set inventory and resumable sparse Max-Cut heuristic sweep.

Unlike 'maxkcut --graphs ALL', this command NEVER calls a dense SDP solver.
All input graphs are processed sequentially with explicit per-graph budgets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .sweep import discover_graphs, inspect_gset, run_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gset-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("reports/gset/full-first-pass"))
    parser.add_argument("--graphs", nargs="+", default=["ALL"],
                        help="ALL for all canonical .txt (or legacy .mat) files, or e.g. G1 G11")
    parser.add_argument("--inspect-only", action="store_true",
                        help="List all instance sizes, weights and parse failures; no optimization")
    parser.add_argument("--restarts", type=int, default=5,
                        help="Small first-pass budget: default 5 local-search starts per graph")
    parser.add_argument("--max-sweeps", type=int, default=20,
                        help="Limit each restart to 20 vertex-order sweeps by default")
    parser.add_argument("--seconds-per-graph", type=float, default=60.,
                        help="Soft wall-time budget, checked during optimization (default 60s)")
    parser.add_argument("--max-graphs", type=int,
                        help="For smoke tests, process the first N sorted graphs only")
    parser.add_argument("--max-n", type=int, default=25000,
                        help="Guard against accidentally huge graphs (default 25000)")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.restarts < 1 or args.max_sweeps < 1:
        parser.error("--restarts and --max-sweeps must be positive")
    if args.seconds_per_graph <= 0 or args.max_n < 2:
        parser.error("--seconds-per-graph must be positive and --max-n >= 2")
    if args.max_graphs is not None and args.max_graphs < 1:
        parser.error("--max-graphs must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")

    if args.inspect_only:
        inventory = inspect_gset(args.gset_dir, args.graphs)
        if args.max_graphs is not None:
            inventory = inventory[:args.max_graphs]
        print(json.dumps(inventory, indent=2))
        return
    run_sweep(
        args.gset_dir, args.output_dir, names=args.graphs,
        restarts=args.restarts, max_sweeps=args.max_sweeps,
        seconds_per_graph=args.seconds_per_graph, max_graphs=args.max_graphs,
        max_n=args.max_n, seed=args.seed,
    )


if __name__ == "__main__":
    main()
