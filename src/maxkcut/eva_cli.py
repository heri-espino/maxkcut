"""Reproduce the EVA vector relaxation and retain its vectors/rounding trials."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import traceback
import sys
from pathlib import Path

import numpy as np
import scipy

from .archive import new_archive, new_run_id, save_method, update_manifest, verify_archive
from .eva import _identity, _prepared_graph, load_eva_factor, solve_eva
from .eva_numba import solve_eva_numba
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


def _full_sdp_comparison(graph: str, input_hash: str, folders, eva_stats):
    """First complete matching full-SDP archive in caller-specified directory order.

    Never select a benchmark on best score: preserve predeclared input ordering.
    Missing or unmatched full SDP cases are not numeric comparisons.
    """
    if isinstance(folders, (str, Path)):
        folders = [Path(folders)]
    issues = []
    for folder in folders:
        folder = Path(folder)
        case_path = folder / "cases" / f"{graph}.json"
        if not case_path.is_file():
            issues.append(f"{folder}: case not found")
            continue
        try:
            case = json.loads(case_path.read_text(encoding="utf8"))
            if case.get("status") != "complete":
                issues.append(f"{folder}: status {case.get('status')}")
                continue
            if case.get("input_sha256") != input_hash:
                issues.append(f"{folder}: graph SHA256 mismatch")
                continue
            source = Path(case["source_jsonl"])
            if not source.is_file():
                attempt = int(case.get("attempts", 1))
                source = (folder / "runs" / graph
                          / f"attempt-{attempt:03d}" / f"{graph}.jsonl")
            if not source.is_file():
                issues.append(f"{folder}: per-trial JSONL unavailable")
                continue
            archive_path = Path(case.get("archive_path", ""))
            if not archive_path.is_dir():
                issues.append(f"{folder}: original partition archive unavailable")
                continue
            verified = verify_archive(archive_path)
            if verified["status"] != "complete" or verified["graph"] != graph:
                issues.append(f"{folder}: SDP archive failed verification")
                continue
            distributions = {name: [] for name in ("FJ", "K-Means2N", "K-MeansNM")}
            rows = []
            for line in source.read_text(encoding="utf8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    if row.get("graph") == graph and row.get("method") in distributions:
                        distributions[row["method"]].append(float(row["cut_weight"]))
                        rows.append(row)
            if not all(distributions.values()):
                issues.append(f"{folder}: missing SDP rounding records")
                continue
            if any(
                len(distributions[method]) != verified["methods"][method]["verified_partitions"]
                for method in distributions
            ):
                issues.append(f"{folder}: incomplete SDP rounding counts")
                continue
            pairs = {}
            for name, values in distributions.items():
                eva = eva_stats[f"{name}_EVA"]
                pairs[name] = {
                    "convex_best": max(values),
                    "convex_mean": float(np.mean(values)),
                    "convex_trials": len(values),
                    "eva_best": eva["best"], "eva_mean": eva["mean"],
                    "eva_trials": eva["restarts"],
                    "best_difference_eva_minus_convex": eva["best"] - max(values),
                }
            return {
                "status": "compared", "graph_sha256": input_hash,
                "convex_source": str(folder),
                "selection_policy": "first complete and archive-verified matching folder",
                "convex_solver_status": case.get("sdp_solver_status"),
                "convex_objective_primal_uncertified":
                    rows[0].get("sdp_objective_primal"),
                "convex_solver_seconds_original":
                    rows[0].get("sdp_original_solve_seconds"),
                "caution": "Different continuous factors; not an isolated rounding comparison",
                "methods": pairs,
            }
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append(f"{folder}: invalid data ({type(exc).__name__}: {exc})")
    return {
        "status": ("graph_mismatch" if issues and
                   all("graph SHA256 mismatch" in issue for issue in issues)
                   else "unavailable"),
        "reason": "; ".join(issues) if issues else "No convex SDP directory configured",
    }

def run_one(path: Path, args):
    backend = getattr(args, "backend", "numpy")
    if backend not in ("numpy", "numba"):
        raise ValueError("Unsupported EVA backend")
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
    case_path = root / "cases" / f"{name}.json"
    factor_path = root / "factors" / f"{name}.npz"
    if case_path.is_file():
        previous = json.loads(case_path.read_text(encoding="utf8"))
        old = previous.get("provenance", {})
        expected_rank = args.rank if args.rank is not None else int(np.ceil(np.sqrt(2 * (n + 1))))
        matches = (
            old.get("optimizer_backend", "numpy") == backend
            and previous.get("graph_sha256") == input_hash
            and int(previous.get("factor_rank", -1)) == expected_rank
            and old.get("seed") == args.seed
            and old.get("cpu_threads_requested") == args.cpu_threads
            and old.get("tolerance") == args.tolerance
            and old.get("rounding_seed") == args.rounding_seed
            and old.get("restarts") == args.restarts
        )
        if not matches:
            raise ValueError(
                f"{name}: saved result has different graph/settings; "
                "use a separate --output-dir to preserve previous experiments"
            )
        factor, factor_meta = load_eva_factor(factor_path)
        archive = Path(previous["archive_path"])
        verified = verify_archive(archive)
        expected_identity = _identity(
            _prepared_graph(graph), expected_rank, args.seed, args.tolerance
        )
        if backend == "numba":
            expected_identity["algorithm"] = "eva-sequential-numba-v1"
        finished = (
            factor_meta.get("identity") == expected_identity
            and all(
                m in verified["methods"]
                and verified["methods"][m]["verified_partitions"] == args.restarts
                for m in METHODS
            )
            and verified["status"] == "complete"
            and factor_meta.get("sweeps") == previous.get("factor_sweeps")
            and factor.shape == (n, expected_rank)
            and (root / "trials" / f"{name}.jsonl").is_file()
        )
        if finished and (factor_meta.get("converged")
                         or factor_meta.get("sweeps", 0) >= args.max_sweeps):
            if args.convex_dir:
                previous["comparison_to_full_sdp"] = _full_sdp_comparison(
                    name, input_hash, args.convex_dir, previous["methods"]
                )
                _write_json(case_path, previous)
            print(f"{name}: verified resume skip (all partitions preserved)", flush=True)
            return previous
        if not finished:
            raise ValueError(f"{name}: existing result archive is incomplete or invalid")
        print(f"{name}: continuing factor optimization to {args.max_sweeps} sweeps", flush=True)
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

    optimizer = solve_eva_numba if backend == "numba" else solve_eva
    with blas_limit(args.cpu_threads):
        vectors, diagnostics = optimizer(
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
        "backend": (
            "EVA sequential compiled Numba float64 (nonconvex vector relaxation)"
            if backend == "numba" else
            "EVA sequential NumPy float64 (nonconvex vector relaxation)"
        ),
        "optimizer_backend": backend,
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
    _write_json(case_path, result)
    if args.push:
        _git_push(root, f"{name} factor and all roundings", allowed_branch=args.push_branch)
    return result



def _write_batch_summary(root: Path, graph_paths):
    """Refresh compact public CSV after each graph, including missing/failures."""
    columns = (
        "graph", "status", "vertices", "edges", "eva_rank", "eva_sweeps",
        "eva_converged", "eva_objective_primal_uncertified", "eva_seconds",
        "eva_FJ_best", "eva_KMeans2N_best", "eva_KMeansNM_best",
        "convex_status", "convex_source", "convex_solver_status",
        "convex_objective_primal_uncertified", "convex_seconds",
        "convex_FJ_best", "convex_KMeans2N_best", "convex_KMeansNM_best",
        "best_cut_eva_minus_convex", "graph_sha256",
    )
    output = root / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for path in graph_paths:
            name = path.stem
            case_path = root / "cases" / f"{name}.json"
            if not case_path.is_file():
                error = root / "errors" / f"{name}.json"
                writer.writerow({"graph": name, "status": "failed" if error.exists() else "pending"})
                continue
            case = json.loads(case_path.read_text(encoding="utf8"))
            methods = case.get("methods", {})
            compare = case.get("comparison_to_full_sdp", {})
            groups = compare.get("methods", {})
            eva_best = max((v["best"] for v in methods.values()), default=None)
            convex_best = max((v["convex_best"] for v in groups.values()), default=None)
            writer.writerow({
                "graph": name, "status": case.get("status"),
                "vertices": case.get("vertices"), "edges": case.get("edges"),
                "eva_rank": case.get("factor_rank"),
                "eva_sweeps": case.get("factor_sweeps"),
                "eva_converged": case.get("status") == "converged",
                "eva_objective_primal_uncertified": case.get("objective_vector_relaxation"),
                "eva_seconds": case.get("factor_elapsed_seconds"),
                "eva_FJ_best": methods.get("FJ_EVA", {}).get("best"),
                "eva_KMeans2N_best": methods.get("K-Means2N_EVA", {}).get("best"),
                "eva_KMeansNM_best": methods.get("K-MeansNM_EVA", {}).get("best"),
                "convex_status": compare.get("status"),
                "convex_source": compare.get("convex_source"),
                "convex_solver_status": compare.get("convex_solver_status"),
                "convex_objective_primal_uncertified":
                    compare.get("convex_objective_primal_uncertified"),
                "convex_seconds": compare.get("convex_solver_seconds_original"),
                "convex_FJ_best": groups.get("FJ", {}).get("convex_best"),
                "convex_KMeans2N_best": groups.get("K-Means2N", {}).get("convex_best"),
                "convex_KMeansNM_best": groups.get("K-MeansNM", {}).get("convex_best"),
                "best_cut_eva_minus_convex": (
                    eva_best - convex_best
                    if eva_best is not None and convex_best is not None else None
                ),
                "graph_sha256": case.get("graph_sha256"),
            })
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=["G11"])
    parser.add_argument("--gset-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-n", type=int, default=2000)
    parser.add_argument("--rank", type=int, help="Embedding dimension; default ceil(sqrt(2(n+1)))")
    parser.add_argument("--backend", choices=("numpy", "numba"), default="numpy",
                        help="Numba compiles the original sequential EVA update; "
                             "use a separate output directory for this backend")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--max-sweeps", type=int, default=10000)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu-threads", type=int, default=1,
                        help="Limit BLAS threads (1 matches official MATLAB single-thread setup)")
    parser.add_argument("--rounding-seed", type=int, default=2026)
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("results/gset/experiment002-eva"))
    parser.add_argument("--convex-dir", type=Path, nargs="+",
                        help="Existing full-SDP batch directories, searched in stated order")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Record per-graph errors and continue to later graphs")
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
    graph_paths = discover_graphs(args.gset_dir, args.graphs)
    graph_paths.sort(key=lambda path: int(path.stem[1:]))
    if args.push and (args.output_dir.parts[:1] == ("reports",)):
        parser.error("Tracked EVA outputs must not be under ignored reports/")
    failures = 0
    for index, path in enumerate(graph_paths, 1):
        print(f"[{index}/{len(graph_paths)}] {path.stem}", flush=True)
        try:
            result = run_one(path, args)
            (args.output_dir / "errors" / f"{path.stem}.json").unlink(missing_ok=True)
            print(f"{result['graph']}: {result['status']} / {result['factor_sweeps']} sweeps", flush=True)
        except (OSError, ValueError, RuntimeError, ArithmeticError, KeyError,
                ImportError) as exc:
            failures += 1
            _write_json(args.output_dir / "errors" / f"{path.stem}.json", {
                "graph": path.stem, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().replace(
                    str(Path.home()), "<HOME>"
                ).replace(str(Path.cwd()), "<REPO>"),
                "command": [str(arg).replace(str(Path.home()), "<HOME>")
                            for arg in sys.argv],
            })
            print(f"{path.stem}: FAILED: {type(exc).__name__}: {exc}", flush=True)
            if args.push:
                _git_push(args.output_dir, f"{path.stem} failure log",
                          allowed_branch=args.push_branch)
            if not args.continue_on_error:
                raise
        _write_batch_summary(args.output_dir, graph_paths)
        if args.push:
            _git_push(args.output_dir, "updated all-graph summary",
                      allowed_branch=args.push_branch)
    print(f"EVA graphs: {len(graph_paths)}, failed: {failures}, summary: "
          f"{args.output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
