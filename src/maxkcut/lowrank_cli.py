"""Optional CUDA (or CPU) low-rank Max-Cut experiment runner.

This uses a NONCONVEX rank-r approximation, NOT CVXPY's full SDP solver.
Its GPU workload consists of sparse graph-vector products and optimizer
updates. The thesis-inspired GW/K-Means rounding runs on the CPU afterward.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .lowrank import torch_device_info
from .lowrank_experiments import benchmark_lowrank
from .sweep import discover_graphs, inspect_gset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gset-dir", type=Path, default=Path("data"))
    parser.add_argument("--graphs", nargs="+", default=["G11"])
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="Default cuda; no silent fallback if PyTorch CUDA is unavailable")
    parser.add_argument("--device-info", action="store_true",
                        help="Inspect chosen PyTorch compute backend without a graph solve")
    parser.add_argument("--inspect-only", action="store_true",
                        help="Inspect selected graphs without solving")
    parser.add_argument("--allow-all", action="store_true",
                        help="Required opt-in for all G-set graphs; beware the compute budget")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--optimizer-starts", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--restarts", type=int, default=20,
                        help="GW/K-Means rounding repetitions per method")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Seed for optimization; changes the factor cache key")
    parser.add_argument("--rounding-seed", type=int,
                        help="Independent rounding seed, defaults to --seed; does not affect factor cache")
    parser.add_argument("--max-n", type=int, default=1000,
                        help="Explicitly raise for larger G-set instances")
    parser.add_argument("--output", type=Path,
                        default=Path("reports/gset/torch-lowrank-runs.jsonl"))
    parser.add_argument("--factor-cache-dir", type=Path,
                        default=Path("reports/gset/lowrank_cache"))
    parser.add_argument("--no-factor-cache", action="store_true")
    parser.add_argument("--no-artifacts", action="store_true")
    parser.add_argument("--artifacts-dir", type=Path)
    args = parser.parse_args()

    if args.device_info:
        print(json.dumps(torch_device_info(args.device), indent=2))
        return
    if args.graphs == ["ALL"] and not args.allow_all and not args.inspect_only:
        parser.error("--graphs ALL requires --allow-all for a substantial compute budget")
    if args.rank < 1 or args.max_n < 2 or args.epochs < 1:
        parser.error("--rank and --epochs must be positive, --max-n >= 2")
    if args.optimizer_starts < 1 or args.restarts < 1 or args.log_every < 1:
        parser.error("--optimizer-starts, --restarts and --log-every must be positive")
    if args.rounding_seed is not None and args.rounding_seed < 0:
        parser.error("--rounding-seed must be nonnegative")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    paths = discover_graphs(args.gset_dir, args.graphs)
    if args.inspect_only:
        print(json.dumps(inspect_gset(args.gset_dir, args.graphs), indent=2))
        return

    benchmark_lowrank(
        paths, args.output, device=args.device, rank=args.rank,
        epochs=args.epochs, optimizer_starts=args.optimizer_starts,
        learning_rate=args.learning_rate, log_every=args.log_every,
        restarts=args.restarts, seed=args.seed,
        rounding_seed=args.rounding_seed, max_n=args.max_n,
        factor_cache_dir=args.factor_cache_dir,
        use_factor_cache=not args.no_factor_cache,
        save_artifacts=not args.no_artifacts,
        artifacts_dir=args.artifacts_dir,
        argv=list(sys.argv),
    )


if __name__ == "__main__":
    main()
