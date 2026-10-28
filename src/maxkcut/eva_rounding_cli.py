"""Reuse archived EVA vectors for independent, append-only rounding experiments.

A completed EVA factor is never optimized again. Existing 20-restart
roundings are left untouched; an extension can add indices [20, 100).
A further extension (or a new algorithm) creates a distinct verified archive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .archive import (
    load_method, new_archive, new_run_id, save_method, update_manifest,
    verify_archive,
)
from .eva import _identity, _prepared_graph, load_eva_factor
from .eva_cli import _git_push, _write_json
from .io import edge_arrays, load_graph
from .local_search import improve_by_flips
from .objective import cut_weight
from .rounding import round_fj, round_spherical_kmeans
from .sweep import discover_graphs


# To add another rounding algorithm, add a named branch in _round_one.
METHODS = (
    "FJ_EVA", "K-Means2N_EVA", "K-MeansNM_EVA",
    "FJ-Flip_EVA", "K-MeansNM-Flip_EVA",
)
BASE_METHODS = METHODS[:3]


def _round_one(vectors, graph, edges, method, graph_number, restart, seed,
               max_flip_sweeps=20):
    """Use the original EVA experiment's per-method/per-restart seed schedule."""
    shared_gw = method in (
        "FJ_EVA", "K-MeansNM_EVA",
        "FJ-Flip_EVA", "K-MeansNM-Flip_EVA",
    )
    rng = np.random.default_rng(np.random.SeedSequence(
        [seed, graph_number, 2, restart, 0 if shared_gw else 1]
    ))
    started = time.monotonic()
    if method in ("FJ_EVA", "FJ-Flip_EVA"):
        labels = round_fj(vectors, 2, rng)
        iterations = 1
    elif method in ("K-Means2N_EVA", "K-MeansNM_EVA", "K-MeansNM-Flip_EVA"):
        antipodal = method != "K-Means2N_EVA"
        labels, iterations = round_spherical_kmeans(
            vectors, 2, rng, antipodal=antipodal,
            edges=edges if antipodal else None,
            keep_best=antipodal,
        )
    else:
        raise ValueError(f"Unknown rounding method {method!r}")

    details = {"rounding_iterations": iterations}
    if method in ("FJ-Flip_EVA", "K-MeansNM-Flip_EVA"):
        before = float(cut_weight(edges, labels))
        # Separate RNG: adding local search never changes the initial
        # GW/k-means direction or the original experimental baseline.
        flips_rng = np.random.default_rng(np.random.SeedSequence(
            [seed, graph_number, 2, restart, 42]
        ))
        labels, diagnostics = improve_by_flips(
            graph, labels, flips_rng, max_sweeps=max_flip_sweeps,
        )
        after = float(cut_weight(edges, labels))
        if after + 1e-8 < before:
            raise AssertionError("Local search must not decrease cut weight")
        details.update({
            "before_flips_cut": before,
            "flip_count": diagnostics["flips"],
            "flip_sweeps": diagnostics["sweeps"],
            "flip_termination": diagnostics["termination"],
        })
    details["rounding_seconds"] = time.monotonic() - started
    return np.asarray(labels, dtype=np.int8), details


def _factor_and_case(path, source):
    graph = load_graph(path)
    graph_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    case_file = source / "cases" / f"{path.stem}.json"
    if not case_file.is_file():
        raise FileNotFoundError(f"No completed EVA case for {path.stem}: {case_file}")
    case = json.loads(case_file.read_text(encoding="utf8"))
    if case.get("graph_sha256") != graph_sha:
        raise ValueError(f"{path.stem}: EVA input graph SHA256 mismatch")
    if case.get("status") not in ("converged", "iteration_limit"):
        raise ValueError(f"{path.stem}: EVA case is not finished")
    factor_file = source / "factors" / f"{path.stem}.npz"
    y, meta = load_eva_factor(factor_file)
    saved = case.get("provenance", {})
    if (y.shape[0] != graph.shape[0] or
            y.shape[1] != case.get("factor_rank") or
            case.get("factor_sweeps") != meta.get("sweeps") or
            meta.get("identity") != _identity(
                _prepared_graph(graph), y.shape[1],
                saved["seed"], saved["tolerance"],
            )):
        raise ValueError(f"{path.stem}: factor/case/graph identity mismatch")
    sha = hashlib.sha256(factor_file.read_bytes()).hexdigest()
    return graph, y, case, graph_sha, sha


def _segment_path(dest, graph_name, factor_sha, method, first, stop):
    label = f"{graph_name}-{factor_sha[:16]}"
    return dest / "segments" / label / method / f"r{first:05d}-{stop:05d}.json"


def _immutable_factor_copy(source, dest, graph_name, factor_sha):
    """Retain exactly the factor used, even if EVA later continues optimizing."""
    original = Path(source) / "factors" / f"{graph_name}.npz"
    target = Path(dest) / "factor-snapshots" / (
        f"{graph_name}-{factor_sha[:16]}.npz"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if hashlib.sha256(target.read_bytes()).hexdigest() != factor_sha:
            raise ValueError("Existing immutable factor snapshot hash mismatch")
        return target
    temporary = target.with_suffix(".npz.tmp")
    try:
        shutil.copyfile(original, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != factor_sha:
            raise ValueError("Factor changed while copying it for the experiment")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _validate_segment(info, graph_sha, factor_sha, seed, flip_sweeps):
    if (info.get("graph_sha256") != graph_sha or
            info.get("factor_sha256") != factor_sha or
            info.get("rounding_seed") != seed or
            info.get("max_flip_sweeps") != flip_sweeps):
        raise ValueError("Saved extension has mismatched graph, factor, or settings")
    report = verify_archive(Path(info["archive_path"]))
    if (report.get("status") != "complete" or
            report.get("graph") != info["graph"] or
            report["methods"][info["method"]]["verified_partitions"]
            != info["stop_restart_exclusive"] - info["start_restart"]):
        raise ValueError("Saved extension archive failed independent verification")
    _, scores = load_method(
        Path(info["archive_path"]) / f"{info['method']}.npz"
    )
    return scores


def _all_scores(path, source, dest, graph_sha, factor_sha, seed, flip_sweeps):
    """Load and independently verify all completed original/extension scores.

    Index collisions are rejected, not silently overwritten.
    """
    counts = {}
    base_case = json.loads(
        (source / "cases" / f"{path.stem}.json").read_text(encoding="utf8")
    )
    original_archive = Path(base_case["archive_path"])
    original = verify_archive(original_archive)
    if (original["status"] != "complete" or original["graph"] != path.stem or
            base_case["graph_sha256"] != graph_sha):
        raise ValueError(f"{path.stem}: base EVA partition archive is not verified")
    base_seed = base_case["provenance"]["rounding_seed"]
    if seed == base_seed:
        for method in BASE_METHODS:
            if method in original["methods"]:
                _, values = load_method(original_archive / f"{method}.npz")
                counts[method] = {i: float(v) for i, v in enumerate(values)}
    root = dest / "segments" / f"{path.stem}-{factor_sha[:16]}"
    for file in sorted(root.glob("*/r*-*.json")):
        info = json.loads(file.read_text(encoding="utf8"))
        values = _validate_segment(info, graph_sha, factor_sha, seed, flip_sweeps)
        method = info["method"]
        scores = counts.setdefault(method, {})
        for index, score in enumerate(values, start=info["start_restart"]):
            if index in scores:
                raise ValueError(
                    f"{path.stem}/{method}: duplicate restart {index} in saved archives"
                )
            scores[index] = float(score)
    return counts


def round_saved_factor(path, *, source, dest, methods, first, restarts,
                       seed=None, max_flip_sweeps=20):
    """Generate one append-only segment per requested method without EVA solves."""
    source, dest = Path(source), Path(dest)
    graph, vectors, case, graph_sha, factor_sha = _factor_and_case(path, source)
    seed = case["provenance"]["rounding_seed"] if seed is None else seed
    edges = edge_arrays(graph)
    count = vectors.shape[0]
    dest.mkdir(parents=True, exist_ok=True)
    if first < 0 or restarts < 1 or seed < 0 or max_flip_sweeps < 1:
        raise ValueError("Invalid rounding range, seed or flip limit")
    if len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("Unknown or duplicated rounding method")
    snapshot = _immutable_factor_copy(source, dest, path.stem, factor_sha)
    stop = first + restarts
    for method in methods:
        segment = _segment_path(dest, path.stem, factor_sha, method, first, stop)
        if segment.is_file():
            info = json.loads(segment.read_text(encoding="utf8"))
            _validate_segment(info, graph_sha, factor_sha, seed, max_flip_sweeps)
            print(f"{path.stem} {method} {first}:{stop}: verified skip", flush=True)
            continue
        # Never overwrite existing original or extension cuts.
        # Refuse ranges that overlap any previously archived scores.
        known = _all_scores(
            path, source, dest, graph_sha, factor_sha, seed, max_flip_sweeps
        )
        overlap = set(range(first, stop)).intersection(known.get(method, {}))
        if overlap:
            raise ValueError(
                f"{path.stem}/{method}: restarts already exist "
                f"(e.g. {min(overlap)}); choose a new, nonoverlapping range"
            )
        labels_list, scores, trial_details = [], [], []
        for restart in range(first, stop):
            labels, details = _round_one(
                vectors, graph, edges, method, int(path.stem[1:]),
                restart, seed, max_flip_sweeps,
            )
            score = float(cut_weight(edges, labels))
            labels_list.append(labels)
            scores.append(score)
            trial_details.append({
                "graph": path.stem, "method": method,
                "restart": restart, "cut_weight": score, **details,
            })
        settings = {
            "backend": "EVA saved-factor rounding; NO EVA optimization",
            "graph_sha256": graph_sha, "factor_sha256": factor_sha,
            "original_factor_sweeps": case["factor_sweeps"],
            "factor_rank": vectors.shape[1],
            "method": method, "rounding_seed": seed,
            "start_restart": first, "stop_restart_exclusive": stop,
            "max_flip_sweeps": max_flip_sweeps,
        }
        archive = new_archive(
            dest / "archives", new_run_id(), path, count, edges, 2,
            settings, jsonl_output=dest / "trials" / path.stem / method,
            argv=sys.argv,
        )
        archive["manifest"]["graph_source"] = str(path)
        archive["manifest"]["results_jsonl"] = (
            f"rounding-extensions/segments/{path.stem}/{method}"
        )
        update_manifest(archive)
        save_method(archive, method, labels_list, scores, k=2)
        update_manifest(archive, status="complete")
        checked = verify_archive(archive["path"])
        if checked["methods"][method]["verified_partitions"] != restarts:
            raise RuntimeError("The new extension archive did not verify")
        segment.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "graph": path.stem, "method": method,
            "graph_sha256": graph_sha, "factor_sha256": factor_sha,
            "rounding_seed": seed, "max_flip_sweeps": max_flip_sweeps,
            "start_restart": first, "stop_restart_exclusive": stop,
            "archive_path": str(archive["path"]),
            "factor_path": str(snapshot),
            "trial_details": trial_details,
        }
        _write_json(segment, record)
        print(
            f"{path.stem} {method} {first}:{stop}: "
            f"saved {restarts} cuts, best={max(scores)}", flush=True,
        )
    combined = _all_scores(
        path, source, dest, graph_sha, factor_sha, seed, max_flip_sweeps
    )
    summary = {
        "graph": path.stem, "graph_sha256": graph_sha,
        "factor_sha256": factor_sha, "factor_snapshot_path": str(snapshot),
        "factor_rank": vectors.shape[1], "factor_sweeps": case["factor_sweeps"],
        "rounding_seed": seed, "max_flip_sweeps": max_flip_sweeps,
        "methods": {
            method: {
                "count": len(samples), "best": max(samples.values()),
                "mean": float(np.mean(list(samples.values()))),
                "median": float(np.median(list(samples.values()))),
                "std": float(np.std(list(samples.values()), ddof=1))
                if len(samples) > 1 else None,
                "restart_indices": sorted(samples),
            }
            for method, samples in combined.items() if samples
        },
    }
    _write_json(dest / "summaries" / f"{path.stem}-{factor_sha[:16]}.json", summary)
    return summary


def _write_collection_summary(dest):
    """Regenerate CSV from committed per-graph summaries; no ephemeral RAM state."""
    dest = Path(dest)
    paths = sorted((dest / "summaries").glob("G*.json"))
    output = dest / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=(
                "graph", "method", "rounding_count", "best_cut",
                "mean_cut", "median_cut", "std_cut", "factor_rank",
                "factor_sweeps", "graph_sha256", "factor_sha256",
            ),
        )
        writer.writeheader()
        for path in paths:
            summary = json.loads(path.read_text(encoding="utf8"))
            for method, stats in sorted(summary["methods"].items()):
                writer.writerow({
                    "graph": summary["graph"], "method": method,
                    "rounding_count": stats["count"],
                    "best_cut": stats["best"], "mean_cut": stats["mean"],
                    "median_cut": stats["median"], "std_cut": stats["std"],
                    "factor_rank": summary["factor_rank"],
                    "factor_sweeps": summary["factor_sweeps"],
                    "graph_sha256": summary["graph_sha256"],
                    "factor_sha256": summary["factor_sha256"],
                })
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=["G11"])
    parser.add_argument("--gset-dir", type=Path, default=Path("data"))
    parser.add_argument("--source-dir", type=Path,
                        default=Path("results/gset/experiment002-eva"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/gset/experiment002-eva/rounding-extensions"))
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(BASE_METHODS))
    parser.add_argument("--start-restart", type=int, default=20)
    parser.add_argument("--restarts", type=int, default=80,
                        help="New additional roundings PER method, not the total")
    parser.add_argument("--rounding-seed", type=int, help="Default: original EVA rounding seed")
    parser.add_argument("--flip-sweeps", type=int, default=20)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--push-branch", help="Required exact experiment branch for push")
    args = parser.parse_args()
    if args.push and not args.push_branch:
        parser.error("--push requires --push-branch")
    if args.output_dir.resolve() == args.source_dir.resolve():
        parser.error("--output-dir must differ from --source-dir")
    paths = discover_graphs(args.gset_dir, args.graphs)
    errors = 0
    for path in paths:
        try:
            round_saved_factor(
                path, source=args.source_dir, dest=args.output_dir,
                methods=args.methods, first=args.start_restart,
                restarts=args.restarts, seed=args.rounding_seed,
                max_flip_sweeps=args.flip_sweeps,
            )
            (args.output_dir / "errors" / f"{path.stem}.json").unlink(missing_ok=True)
        except (OSError, ValueError, KeyError, RuntimeError, AssertionError) as exc:
            errors += 1
            _write_json(args.output_dir / "errors" / f"{path.stem}.json", {
                "graph": path.stem, "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"{path.stem}: {type(exc).__name__}: {exc}", flush=True)
            if not args.continue_on_error:
                if args.push:
                    _git_push(args.output_dir, f"{path.stem} rounding failure",
                              allowed_branch=args.push_branch)
                raise
        _write_collection_summary(args.output_dir)
        if args.push:
            _git_push(args.output_dir, f"{path.stem} reused-factor rounding",
                      allowed_branch=args.push_branch)
    print(f"Rounding extensions: {len(paths)} graphs; errors={errors}; "
          f"{args.output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
