"""Bench CPU NumPy versus compiled sequential EVA, preserving measured runs.

No saved existing EVA factor or rounding archive is touched.
Reference and compiled variants receive identical input graph, seed,
rank, iteration limit and convergence tolerance. JIT compilation is
reported separately from timed numerical optimization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .eva import solve_eva
from .eva_numba import solve_eva_numba
from .eva_cli import _git_push, _write_json
from .io import load_graph
from .sweep import discover_graphs


def compare_backends(graph_path: Path, *, seed=2026, tolerance=0.001,
                     rank=None, max_sweeps=10000):
    graph = load_graph(graph_path)
    reference_y, reference = solve_eva(
        graph, seed=seed, tolerance=tolerance, rank=rank,
        max_sweeps=max_sweeps,
    )
    compiled_y, compiled = solve_eva_numba(
        graph, seed=seed, tolerance=tolerance, rank=rank,
        max_sweeps=max_sweeps,
    )
    ref_secs = float(reference["elapsed_seconds"])
    fast_secs = float(compiled["elapsed_seconds"])
    max_vector_difference = float(np.max(np.abs(reference_y - compiled_y)))
    objective_difference = float(compiled["objective"] - reference["objective"])
    return {
        "graph": graph_path.stem, "n": int(graph.shape[0]),
        "rank": int(reference_y.shape[1]),
        "graph_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        "seed": seed, "tolerance": tolerance, "max_sweeps": max_sweeps,
        "reference_seconds": ref_secs, "compiled_seconds": fast_secs,
        "compile_seconds_excluded_from_compiled_seconds":
            float(compiled.get("compile_seconds", 0.0)),
        "speedup_excluding_compilation":
            (ref_secs / fast_secs) if fast_secs > 0 else None,
        "reference_objective": float(reference["objective"]),
        "compiled_objective": float(compiled["objective"]),
        "objective_difference": objective_difference,
        "max_abs_vector_difference": max_vector_difference,
        "reference_sweeps": int(reference["sweeps"]),
        "compiled_sweeps": int(compiled["sweeps"]),
        "reference_converged": bool(reference["converged"]),
        "compiled_converged": bool(compiled["converged"]),
        "vector_difference_caution":
            "Same equation and seed, floating point reduction order may differ",
    }


def _write_summary(root):
    cases = sorted((root / "cases").glob("G*.json"))
    fields = (
        "graph", "n", "rank", "reference_seconds", "compiled_seconds",
        "compile_seconds_excluded_from_compiled_seconds",
        "speedup_excluding_compilation", "reference_objective",
        "compiled_objective", "objective_difference",
        "max_abs_vector_difference", "reference_sweeps",
        "compiled_sweeps", "reference_converged", "compiled_converged",
        "graph_sha256",
    )
    (root / "summary.csv").parent.mkdir(parents=True, exist_ok=True)
    tmp = root / "summary.csv.tmp"
    with tmp.open("w", newline="", encoding="utf8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for file in cases:
            row = json.loads(file.read_text(encoding="utf8"))
            writer.writerow({key: row.get(key) for key in fields})
    tmp.replace(root / "summary.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=["G1", "G11", "G55"])
    parser.add_argument("--gset-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/gset/experiment003-eva-perf"))
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=0.001)
    parser.add_argument("--max-sweeps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--push-branch")
    args = parser.parse_args()
    if args.push and not args.push_branch:
        parser.error("An explicit --push-branch is required for --push")
    for path in discover_graphs(args.gset_dir, args.graphs):
        print(f"Benchmark {path.stem}: NumPy and Numba sequential EVA", flush=True)
        metrics = compare_backends(
            path, seed=args.seed, tolerance=args.tolerance,
            max_sweeps=args.max_sweeps, rank=args.rank,
        )
        _write_json(
            args.output_dir / "cases" / f"{path.stem}.json", metrics,
        )
        _write_summary(args.output_dir)
        print(
            f"{path.stem}: numpy={metrics['reference_seconds']:.3f}s; "
            f"numba={metrics['compiled_seconds']:.3f}s; "
            f"speedup={metrics['speedup_excluding_compilation']:.2f}x; "
            f"objective delta={metrics['objective_difference']:.8g}",
            flush=True,
        )
        if args.push:
            _git_push(
                args.output_dir, f"{path.stem} sequential EVA speed benchmark",
                allowed_branch=args.push_branch,
            )


if __name__ == "__main__":
    main()
