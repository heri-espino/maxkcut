"""Reproduce the EVA vector relaxation and retain its vectors/rounding trials."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy

from .archive import new_archive, new_run_id, save_method, update_manifest
from .eva import load_eva_factor, solve_eva
from .hardware import blas_limit
from .io import edge_arrays, load_graph
from .objective import cut_weight
from .rounding import round_fj, round_spherical_kmeans
from .sweep import discover_graphs


METHODS = ("FJ_EVA", "K-Means2N_EVA", "K-MeansNM_EVA")


def _write_json(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf8")
    temp.replace(path)


def _git_push(root: Path, label: str, *, allowed_branch: str):
    """Non-force opt-in push, restricted to the explicit experiment branch."""
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip()
    if branch != allowed_branch:
        raise RuntimeError(
            f"Refusing Git push: current branch {branch!r} is not {allowed_branch!r}"
        )
    rel = root.resolve().relative_to(Path.cwd().resolve())
    subprocess.run(["git", "add", "--", str(rel)], check=True)
    diff = subprocess.run(["git", "diff", "--cached", "--quiet", "--", str(rel)])
    if diff.returncode == 1:
        subprocess.run([
            "git", "commit", "-m", f"experiment002: {label}", "--", str(rel)
        ], check=True)
    elif diff.returncode != 0:
        raise RuntimeError("Git could not inspect staged results")
    subprocess.run(["git", "push", "origin", f"HEAD:{branch}"], check=True)


def _full_sdp_comparison(graph: str, input_hash: str, folder: Path, eva_stats):
    """Compare verified-identity method scores, not unmatched SDP optima."""
    case_path = folder / "cases" / f"{graph}.json"
    if not case_path.exists():
        return {"status": "unavailable", "reason": "No full SDP case"}
    case = json.loads(case_path.read_text(encoding="utf8"))
    if case.get("status") != "complete":
        return {"status": "unavailable", "reason": f"Full SDP {case.get('status')}"}
    if case.get("input_sha256") != input_hash:
        return {"status": "graph_mismatch", "reason": "Different input graph SHA256"}
    source = Path(case["source_jsonl"])
    if not source.exists():
        return {"status": "unavailable", "reason": "Full SDP trial JSONL missing"}
    distributions = {name: [] for name in ("FJ", "K-Means2N", "K-MeansNM")}
    for line in source.read_text(encoding="utf8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("graph") == graph and row.get("method") in distributions:
                distributions[row["method"]].append(float(row["cut_weight"]))
    if not all(distributions.values()):
        return {"status": "unavailable", "reason": "Incomplete SDP rounding records"}
    pairs = {}
    for name, values in distributions.items():
        eva = eva_stats[f"{name}_EVA"]
        pairs[name] = {
            "convex_best": max(values), "convex_mean": float(np.mean(values)),
            "convex_trials": len(values), "eva_best": eva["best"],
            "eva_mean": eva["mean"], "eva_trials": eva["restarts"],
            "best_difference_eva_minus_convex": eva["best"] - max(values),
        }
    return {
        "status": "compared", "graph_sha256": input_hash,
        "convex_solver_status": case.get("sdp_solver_status"),
        "caution": "Different continuous factors; not an isolated rounding comparison",
        "methods": pairs,
    }


def run_one(path: Path, args):
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    name = path.stem
    graph = load_graph(path)
    n = graph.shape[0]
    if n > args.max_n:
        raise ValueError(f"{name}: n={n} exceeds --max-n={args.max_n}")
    if args.push and not args.push_branch:
        raise ValueError("--push requires --push-branch for explicit branch authorization")
    input_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    factor_path = root / "factors" / f"{name}.npz"
    log_path = root / "logs" / f"{name}-sweeps.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # The sweep log may extend past the last complete checkpoint if a VM
    # was interrupted. Replay only states retained by the factor.
    if factor_path.exists() and log_path.exists():
        _, previous = load_eva_factor(factor_path)
        completed = int(previous["sweeps"])
        kept = []
        for line in log_path.read_text(encoding="utf8").splitlines():
            if line.strip() and json.loads(line)["sweep"] <= completed:
                kept.append(line)
        log_path.write_text(
            chr(10).join(kept) + (chr(10) if kept else ""), encoding="utf8"
        )
    print(f"{name}: EVA n={n}; storing factor at {factor_path}", flush=True)

    def callback(state):
        with log_path.open("a", encoding="utf8") as out:
            out.write(json.dumps({
                "graph": name, "sweep": state["sweeps"],
                "objective": state["objective"], "max_angle": state["max_angle"],
                "elapsed_seconds": state["elapsed_seconds"],
            }) + "\n")
        if (args.push and args.push_every
                and state["sweeps"] % args.push_every == 0):
            _git_push(root, f"{name} sweep {state['sweeps']}", allowed_branch=args.push_branch)

    with blas_limit(args.cpu_threads):
        vectors, diagnostics = solve_eva(
            graph, rank=args.rank, tolerance=args.tolerance,
            max_sweeps=args.max_sweeps, seed=args.seed,
            checkpoint_path=factor_path, checkpoint_every=args.checkpoint_every,
            on_sweep=callback,
        )
    _, persisted = load_eva_factor(factor_path)
    if persisted["sweeps"] != diagnostics["sweeps"]:
        raise RuntimeError("Factor checkpoint is not synchronized with EVA state")
    edges = edge_arrays(graph)
    continuous_score = diagnostics["objective"]
    actual_score = float(0.5 * np.sum(
        edges[2] * (1 - np.einsum(
            "ij,ij->i", vectors[edges[0]], vectors[edges[1]]
        ))
    ))
    if not np.isclose(continuous_score, actual_score, atol=1e-6, rtol=1e-9):
        raise RuntimeError("Inconsistent EVA continuous objective")

    provenance = {
        "backend": "EVA sequential NumPy float64 (nonconvex vector relaxation)",
        "input_sha256": input_hash, "input_file": str(path),
        "code_source": "https://github.com/b3r8/EVA/blob/main/EVA.m",
        "paper_doi": "10.1016/j.rico.2026.100759",
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "machine": platform.platform(), "command": sys.argv,
        "rank": int(vectors.shape[1]), "seed": args.seed,
        "cpu_threads_requested": args.cpu_threads,
        "tolerance": args.tolerance, "max_sweeps": args.max_sweeps,
        "restarts": args.restarts, "rounding_seed": args.rounding_seed,
    }
    archive = new_archive(
        root / "archives", new_run_id(), path, n, edges, 2, provenance,
        jsonl_output=root / "trials" / f"{name}.jsonl", argv=sys.argv,
    )
    # Keep user-specific absolute paths out of publicly tracked manifests.
    archive["manifest"]["graph_source"] = str(path)
    archive["manifest"]["results_jsonl"] = str(root / "trials" / f"{name}.jsonl")
    update_manifest(archive)
    trial_rows = []
    stats = {}
    for method in METHODS:
        labels_list, scores = [], []
        for rep in range(args.restarts):
            rng = np.random.default_rng(np.random.SeedSequence([
                args.rounding_seed, int(name[1:]), 2, rep,
                0 if method in ("FJ_EVA", "K-MeansNM_EVA") else 1
            ]))
            if method == "FJ_EVA":
                labels = round_fj(vectors, 2, rng)
            else:
                labels, _ = round_spherical_kmeans(
                    vectors, 2, rng,
                    antipodal=(method == "K-MeansNM_EVA"),
                    edges=edges if method == "K-MeansNM_EVA" else None,
                    keep_best=(method == "K-MeansNM_EVA"),
                )
            score = float(cut_weight(edges, labels))
            labels_list.append(labels)
            scores.append(score)
            trial_rows.append({
                "graph": name, "method": method, "restart": rep,
                "cut_weight": score, "factor_sweeps": diagnostics["sweeps"],
                "factor_rank": vectors.shape[1],
                "factor_converged": diagnostics["converged"],
                "archive_run_id": archive["manifest"]["run_id"],
            })
        save_method(archive, method, labels_list, scores, k=2)
        stats[method] = {
            "best": float(np.max(scores)), "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
            "restarts": len(scores),
        }
        print(f"  {method}: best={stats[method]['best']}; mean={stats[method]['mean']:.3f}", flush=True)
    update_manifest(archive, status="complete", solver_info={
        "solver": "EVA-vector", **diagnostics,
    })
    trials_path = root / "trials" / f"{name}.jsonl"
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    with trials_path.open("w", encoding="utf8") as out:
        for row in trial_rows:
            out.write(json.dumps(row) + "\n")
    comparisons = _full_sdp_comparison(
        name, input_hash, args.convex_dir, stats
    ) if args.convex_dir else {"status": "not_requested"}
    result = {
        "graph": name, "vertices": n, "edges": len(edges[0]),
        "status": "converged" if diagnostics["converged"] else "iteration_limit",
        "objective_vector_relaxation": continuous_score,
        "factor_path": str(factor_path), "archive_path": str(archive["path"]),
        "factor_sweeps": diagnostics["sweeps"],
        "factor_rank": vectors.shape[1],
        "factor_max_angle": diagnostics["max_angle"],
        "factor_elapsed_seconds": diagnostics["elapsed_seconds"],
        "graph_sha256": input_hash, "methods": stats,
        "comparison_to_full_sdp": comparisons,
        "provenance": provenance,
    }
    _write_json(root / "cases" / f"{name}.json", result)
    if args.push:
        _git_push(root, f"{name} factor and all roundings", allowed_branch=args.push_branch)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=["G11"])
    parser.add_argument("--gset-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-n", type=int, default=2000)
    parser.add_argument("--rank", type=int, help="Embedding dimension; default ceil(sqrt(2(n+1)))")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--max-sweeps", type=int, default=10000)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu-threads", type=int, default=1,
                        help="Limit BLAS threads (1 matches official MATLAB single-thread setup)")
    parser.add_argument("--rounding-seed", type=int, default=2026)
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("results/gset/experiment002-eva"))
    parser.add_argument("--convex-dir", type=Path, help="Existing full-SDP batch output for graph-verified comparison")
    parser.add_argument("--push", action="store_true", help="Commit and push checkpoints/results; explicit branch only")
    parser.add_argument("--push-branch", help="Required exact branch name when --push is used")
    parser.add_argument("--push-every", type=int, default=0, help="Push checkpoint every N sweeps; requires --push and N divisible by checkpoint interval")
    args = parser.parse_args()
    if (args.restarts < 1 or args.cpu_threads < 1 or args.max_n < 2 or args.seed < 0
            or args.rounding_seed < 0):
        parser.error("Invalid positive counts or nonnegative seeds")
    if args.push_every < 0 or (args.push_every and not args.push):
        parser.error("--push-every requires --push")
    if args.push_every and args.push_every % args.checkpoint_every != 0:
        parser.error("--push-every must be a multiple of --checkpoint-every")
    if args.push and not args.push_branch:
        parser.error("--push needs explicit --push-branch")
    for path in discover_graphs(args.gset_dir, args.graphs):
        result = run_one(path, args)
        print(f"{result['graph']}: {result['status']} / {result['factor_sweeps']} sweeps", flush=True)


if __name__ == "__main__":
    main()
