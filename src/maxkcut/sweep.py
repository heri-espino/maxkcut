"""Resumable first-pass sweep over checked-in G-set graphs.

Uses a *sparse local-search baseline* only: it deliberately does NOT try to
solve 71 potentially enormous dense SDPs. Original graphs, all successful
partitions, timings, seeds, per-case manifest and a sortable summary survive.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import time
from uuid import uuid4

import numpy as np

from .archive import (
    load_method, new_archive, new_run_id, save_method, update_manifest,
    verify_archive,
)
from .comparison import write_comparison
from .io import edge_arrays, load_graph
from .local_search import random_flip_restart


SWEEP_VERSION = 1
METHOD = "random_flip_local_search"


def _graph_number(path: Path) -> int:
    match = re.fullmatch(r"G(\d+)\.(?:txt|mat)", path.name)
    if match is None:
        raise ValueError(f"Expected G<number>.txt or .mat, received {path}")
    return int(match.group(1))


def discover_graphs(directory: Path, names: list[str] | None = None) -> list[Path]:
    """Find G-set graphs, using canonical TXT over MAT for duplicate IDs.

    Explicit .txt or .mat names override the extension preference. The
    old G-Set/*.mat directory remains supported with --gset-dir G-Set.
    """
    directory = Path(directory)
    if names is None or names == ["ALL"]:
        by_number = {}
        for path in directory.iterdir() if directory.is_dir() else []:
            if not re.fullmatch(r"G\d+\.(?:txt|mat)", path.name):
                continue
            number = _graph_number(path)
            prior = by_number.get(number)
            if prior is None or (
                path.suffix.lower() == ".txt" and prior.suffix.lower() == ".mat"
            ):
                by_number[number] = path
        paths = list(by_number.values())
    else:
        paths = []
        for name in names:
            provided = Path(name)
            if provided.name != name:
                raise ValueError(f"Graph name must not include directories: {name}")
            if provided.suffix:
                candidate = directory / name
            else:
                txt = directory / f"{name}.txt"
                mat = directory / f"{name}.mat"
                candidate = txt if txt.exists() else mat
            paths.append(candidate)
    if not paths:
        if not directory.is_dir():
            raise FileNotFoundError(
                f"G-set directory does not exist: {directory}. "
                "Synchronize your local repository (git pull --ff-only origin main), "
                "or pass --gset-dir PATH to the actual dataset."
            )
        raise FileNotFoundError(
            f"No G-set G<number>.txt or G<number>.mat files found in {directory}. "
            "Check the directory contents and your local Git checkout."
        )
    for path in paths:
        _graph_number(path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing graph file: {path}")
    if len(set(paths)) != len(paths) or len(set(map(_graph_number, paths))) != len(paths):
        raise ValueError("Repeated graph IDs are not allowed")
    return sorted(paths, key=_graph_number)


def inspect_gset(directory: Path, names: list[str] | None = None) -> list[dict]:
    """Report graph metadata without optimization or producing files."""
    results = []
    for path in discover_graphs(directory, names):
        try:
            graph = load_graph(path)
            edges = edge_arrays(graph)
            results.append({
                "graph": path.stem, "n": int(graph.shape[0]),
                "m": int(len(edges[0])), "weight_min": float(np.min(edges[2])),
                "weight_max": float(np.max(edges[2])), "status": "ok"
            })
        except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
            results.append({
                "graph": path.stem, "status": "error",
                "error": f"{type(error).__name__}: {error}"
            })
    return results


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_trials_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf8") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_summary(output_dir: Path) -> None:
    """Rebuild table from one record per graph; safe after interrupted batches."""
    rows = []
    for file in (output_dir / "graphs").glob("G*.json"):
        try:
            data = json.loads(file.read_text(encoding="utf8"))
        except (OSError, ValueError):
            continue
        rows.append(data)
    rows.sort(key=lambda entry: int(entry["graph"][1:]))
    fields = [
        "graph", "n", "m", "status", "completed_restarts",
        "requested_restarts", "best_cut", "mean_cut", "median_cut",
        "sample_std_cut", "total_seconds", "input_sha256", "archive_path",
        "error",
    ]
    target = output_dir / "summary.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(target)
        write_comparison(target)
    finally:
        temporary.unlink(missing_ok=True)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _settings(seed: int, restarts: int, max_sweeps: int,
              seconds_per_graph: float) -> dict:
    return {
        "algorithm": METHOD,
        "algorithm_version": SWEEP_VERSION,
        "seed": int(seed),
        "restarts": int(restarts),
        "max_sweeps": int(max_sweeps),
        "seconds_per_graph": float(seconds_per_graph),
        "k": 2,
        "sdp_used": False,
    }


def run_sweep(gset_dir: Path, output_dir: Path, *,
              names: list[str] | None = None,
              max_graphs: int | None = None,
              restarts: int = 5,
              max_sweeps: int = 20,
              seconds_per_graph: float = 60.0,
              seed: int = 2026,
              max_n: int = 25000) -> dict:
    """Perform a bounded, resumable sparse heuristic baseline for all graphs.

    Completed graph results with identical settings and unchanged input file
    checksums are skipped after verifying the archived partitions. Failed or
    budget-limited graph cases are recorded and later reattempted only if the
    user changes configuration or deletes that case record. A partially
    completed graph is not mislabeled as five successful restarts.
    """
    if restarts < 1 or max_sweeps < 1 or seconds_per_graph <= 0 or max_n < 2:
        raise ValueError("Invalid nonpositive sweep budget")
    paths = discover_graphs(gset_dir, names)
    if max_graphs is not None:
        if max_graphs < 1:
            raise ValueError("max_graphs must be positive")
        paths = paths[:max_graphs]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = _settings(seed, restarts, max_sweeps, seconds_per_graph)

    plan_path = output_dir / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf8"))
        if plan != settings:
            raise ValueError(
                "Sweep settings differ from plan.json in this directory. "
                "Use a fresh --output-dir for a different experiment budget."
            )
    else:
        _write_json_atomic(plan_path, settings)

    counts = {"completed": 0, "budget_limited": 0, "skipped": 0, "failed": 0}
    for path in paths:
        n_graph = path.stem
        source_hash = _hash_file(path)
        case = output_dir / "graphs" / f"{n_graph}.json"
        if case.exists():
            previous = json.loads(case.read_text(encoding="utf8"))
            if (previous.get("input_sha256") == source_hash
                    and previous.get("settings") == settings
                    and previous.get("status") in ("complete", "budget_limited")
                    and previous.get("archive_path")
                    and Path(previous["archive_path"]).is_dir()):
                try:
                    verified = verify_archive(previous["archive_path"])
                    if (verified["status"] == previous["status"]
                            and sum(
                                info["verified_partitions"]
                                for info in verified["methods"].values()
                            ) == previous["completed_restarts"]):
                        counts["skipped"] += 1
                        print(f"{n_graph}: resume skip ({previous['status']})", flush=True)
                        continue
                except (OSError, ValueError, KeyError):
                    pass

        started = time.monotonic()
        archive = None
        report = {
            "graph": n_graph,
            "input_sha256": source_hash,
            "settings": settings,
            "requested_restarts": restarts,
            "completed_restarts": 0,
            "status": "started",
            "error": None,
        }
        try:
            graph = load_graph(path)
            n = graph.shape[0]
            if n > max_n:
                raise ValueError(f"n={n} exceeds --max-n={max_n}")
            edges = edge_arrays(graph)
            report.update(n=int(n), m=int(len(edges[0])))
            archive = new_archive(
                output_dir / "artifacts", new_run_id(), path, n, edges, 2,
                settings, jsonl_output=output_dir / "trials" / f"{n_graph}.jsonl",
                argv=None,
            )
            update_manifest(
                archive, algorithm_family="sparse_heuristic",
                solver_info=None, sdp_used=False,
            )
            deadline = started + seconds_per_graph
            labels_by_restart = []
            scores = []
            trials = []
            for restart in range(restarts):
                if restart and time.monotonic() >= deadline:
                    break
                # Deterministic restart seeds independent of graph iteration
                # order, other graphs' failures and resume behavior.
                child = np.random.SeedSequence([seed, _graph_number(path), restart])
                trial_seed = int(child.generate_state(1, dtype=np.uint64)[0])
                labels, info = random_flip_restart(
                    graph, seed=trial_seed, max_sweeps=max_sweeps,
                    deadline=deadline,
                )
                labels_by_restart.append(labels)
                scores.append(info["cut_weight"])
                trials.append({
                    "graph": n_graph,
                    "k": 2,
                    "method": METHOD,
                    "restart": restart,
                    "restart_seed": trial_seed,
                    "cut_weight": info["cut_weight"],
                    "initial_cut": info["initial_cut"],
                    "flips": info["flips"],
                    "sweeps": info["sweeps"],
                    "termination": info["termination"],
                    "rounding_seconds": info["seconds"],
                    "sdp_used": False,
                    "artifact_run_id": archive["manifest"]["run_id"],
                    "partition_index": restart,
                    "partition_archive": str(archive["path"] / f"{METHOD}.npz"),
                })
                if info["termination"] == "time_budget":
                    break

            if not scores:
                raise RuntimeError("No heuristic restart completed")
            summary = save_method(
                archive, METHOD, labels_by_restart, scores, k=2
            )
            status = (
                "complete" if len(scores) == restarts
                and all(t["termination"] != "time_budget" for t in trials)
                else "budget_limited"
            )
            update_manifest(
                archive, status=status,
                completed_restarts=len(scores),
                total_seconds=time.monotonic() - started,
            )
            report.update(
                status=status, completed_restarts=len(scores),
                best_cut=summary["best_cut"],
                mean_cut=summary["mean_cut"],
                median_cut=summary["median_cut"],
                sample_std_cut=summary["sample_std_cut"],
                total_seconds=time.monotonic() - started,
                archive_path=str(archive["path"].resolve()),
            )
            _write_trials_atomic(output_dir / "trials" / f"{n_graph}.jsonl", trials)
            # Per-case manifests use status="complete" while the public
            # aggregate summary deliberately uses the key "completed".
            # Do not turn a successful persisted run into a failed one
            # merely because the display counter has a different name.
            counts["completed" if status == "complete" else "budget_limited"] += 1
            print(
                f"{n_graph}: {status}, best={summary['best_cut']}, "
                f"{len(scores)}/{restarts} restarts "
                f"({report['total_seconds']:.2f}s)", flush=True
            )
        except Exception as exc:
            # This is a batch runner: one graph's malformed input or memory
            # failure should not silently suppress subsequent graph attempts.
            report.update(
                status="failed", error=f"{type(exc).__name__}: {exc}",
                total_seconds=time.monotonic() - started,
            )
            if archive is not None:
                update_manifest(archive, status="failed", error=report["error"])
            counts["failed"] += 1
            print(f"{n_graph}: FAILED: {report['error']}", flush=True)

        _write_json_atomic(case, report)
        _write_summary(output_dir)

    _write_summary(output_dir)
    result = {"graphs_requested": len(paths), **counts,
              "comparison_csv": str((output_dir / "comparison.csv").resolve()),
              "summary_csv": str((output_dir / "summary.csv").resolve())}
    print(json.dumps(result, indent=2), flush=True)
    return result
