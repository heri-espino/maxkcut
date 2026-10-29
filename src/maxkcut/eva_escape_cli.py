"""Experiment010: parallel KL-style and Tabu refinement of SAVED EVA roundings.

Read independently verified bit-packed partitions from experiment008
rounding-extension segments. Select the same top-scoring K-MeansNM-Flip
warm starts for BOTH methods, then run independent KL-style and Tabu
search tasks concurrently in separate Windows-safe processes.

NO EVA optimization, no SDP solves, no new rounding, no MST. Every
completed chunk is an immutable, independently verified archive
containing its ORIGINAL weighted graph, ALL refined binary partitions,
scores, source indices, seeds, and per-run diagnostics. Jobs are fully
resumable by settings/input fingerprint. Commit/push once at end.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from .archive import (
    _file_sha256, load_method, new_archive, new_run_id, save_method,
    update_manifest, verify_archive,
)
from .eva_cli import _git_push, _write_json
from .eva_escape import METHODS, optimize_kl, optimize_tabu
from .io import edge_arrays, load_graph
from .objective import cut_weight
from .sweep import discover_graphs


# Graph objects are initialized ONCE per Windows worker. Neither
# optimization kernel nor the archive worker relies on fork semantics.
_WORK_GRAPH = None
_WORK_EDGES = None


def _init_worker(graph_path):
    global _WORK_GRAPH, _WORK_EDGES
    _WORK_GRAPH = load_graph(Path(graph_path))
    _WORK_EDGES = edge_arrays(_WORK_GRAPH)


def _read_initials(graph_path, directory, method, sha_prefix=None):
    """Verify selected source segment archives and recover restart labels."""
    graph_path, directory = Path(graph_path), Path(directory)
    sha = _file_sha256(graph_path)
    pattern = f"{graph_path.stem}-*"
    roots = sorted((directory / "segments").glob(pattern))
    available = []
    for root in roots:
        for segment in sorted((root / method).glob("r*-*.json")):
            record = json.loads(segment.read_text(encoding="utf8"))
            if sha_prefix and not record["factor_sha256"].startswith(sha_prefix):
                continue
            available.append((segment, record))
    if not available:
        raise FileNotFoundError(
            f"No archived {graph_path.stem}/{method} partitions in "
            f"{directory / 'segments'}. Make sure experiment008 results "
            "were committed/pushed and merged into the current branch."
        )
    factor_set = {record["factor_sha256"] for _, record in available}
    if len(factor_set) != 1:
        raise ValueError(
            f"{graph_path.stem} has {len(factor_set)} EVA factor versions; "
            "select --factor-sha-prefix explicitly"
        )
    start_parts = {}
    provenance = []
    expected_seed = set()
    for segment, record in available:
        if (record.get("graph") != graph_path.stem
                or record.get("method") != method
                or record.get("graph_sha256") != sha):
            raise ValueError("Source segment graph or rounding method mismatch")
        expected_seed.add(record["rounding_seed"])
        archive_path = Path(record["archive_path"])
        verified = verify_archive(archive_path)
        if (verified.get("status") != "complete"
                or verified.get("graph_sha256") != sha):
            raise RuntimeError(f"Cannot verify source archive: {archive_path}")
        labels, scores = load_method(archive_path / f"{method}.npz")
        first, stop = record["start_restart"], record["stop_restart_exclusive"]
        if (stop-first != len(scores)
                or verified["methods"][method]["verified_partitions"] != len(scores)):
            raise RuntimeError(f"Source restarts/partitions mismatch: {segment}")
        provenance.append({
            "segment_file": str(segment),
            "segment_sha256": _file_sha256(segment),
            "archive_path": str(archive_path),
            "partition_npz_sha256": _file_sha256(archive_path / f"{method}.npz"),
            "rounding_seed": record["rounding_seed"],
            "max_flip_sweeps": record["max_flip_sweeps"],
            "range": [first, stop],
        })
        for offset, (partition, score) in enumerate(zip(labels, scores)):
            restart_index = first + offset
            if restart_index in start_parts:
                raise ValueError("Overlapping source restart indices")
            start_parts[restart_index] = (partition.astype(np.int8), float(score))
    if len(expected_seed) != 1:
        raise ValueError("Source segments disagree on rounding seed")
    if not start_parts:
        raise ValueError("No saved rounding partitions found")
    return next(iter(factor_set)), start_parts, provenance


def _select_starts(all_starts, top_starts):
    ranked = sorted(all_starts, key=lambda i: (-all_starts[i][1], i))
    if top_starts < 0:
        raise ValueError("top_starts must be >= 0 (0 means all)")
    chosen = ranked if top_starts == 0 else ranked[:top_starts]
    if not chosen:
        raise ValueError("No source partitions selected")
    return chosen


def _case_identity(method, chunk, graph_sha, factor_sha, source_method,
                   seed, kl_passes, tabu_steps, tabu_tenure, tabu_stagnation,
                   source_provenance):
    input_ids = [x["restart"] for x in chunk]
    source_files = sorted(x["partition_npz_sha256"] for x in source_provenance)
    config = {
        "version": "experiment010-kl-tabu-v1",
        "graph_sha256": graph_sha,
        "factor_sha256": factor_sha,
        "source_method": source_method,
        "source_partition_npz_sha256": source_files,
        "method": method,
        "source_restarts": input_ids,
        "source_starting_scores": [x["initial_cut"] for x in chunk],
        "seed": seed,
        "kl_passes": kl_passes if method == "KL" else None,
        "tabu_steps": tabu_steps if method == "Tabu" else None,
        "tabu_tenure": tabu_tenure if method == "Tabu" else None,
        "tabu_stagnation": tabu_stagnation if method == "Tabu" else None,
    }
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf8")
    ).hexdigest()[:20]
    return fingerprint, config


def _verify_case(record, config):
    if (record.get("status") != "complete"
            or record.get("settings") != config):
        raise ValueError("Saved KL/Tabu chunk has different input/settings")
    verified = verify_archive(record["archive_path"])
    if (verified["status"] != "complete"
            or verified["graph_sha256"] != config["graph_sha256"]
            or verified["methods"][record["archive_method"]][
                "verified_partitions"] != len(record["trials"])):
        raise RuntimeError("Saved KL/Tabu result partition verification failed")
    _, scores = load_method(
        Path(record["archive_path"]) / f"{record['archive_method']}.npz"
    )
    expected = np.asarray([trial["final_cut"] for trial in record["trials"]])
    if (not np.allclose(scores, expected, rtol=0, atol=1e-8)
            or not np.all(expected + 1e-8 >= np.asarray(
                config["source_starting_scores"],
            ))):
        raise RuntimeError("Saved KL/Tabu scores or monotonicity invalid")
    return record


def _run_chunk(job):
    method = job["method"]
    config = job["settings"]
    case_path = Path(job["case_path"])
    if case_path.is_file():
        record = json.loads(case_path.read_text(encoding="utf8"))
        return _verify_case(record, config), True
    if _WORK_GRAPH is None:
        raise RuntimeError("Graph worker not initialized")
    if _file_sha256(Path(job["graph_path"])) != config["graph_sha256"]:
        raise ValueError("Graph file changed while solving")
    archive_method = (
        f"{config['source_method']}-{method}_EVA"
    )
    trials = []
    refined = []
    scores = []
    for start in job["starts"]:
        restart = start["restart"]
        original = np.asarray(start["labels"], dtype=np.int8)
        initial = float(start["initial_cut"])
        rescored = float(cut_weight(_WORK_EDGES, original))
        if not np.isclose(rescored, initial, rtol=0, atol=1e-8):
            raise ValueError(f"Source restart {restart} fails independent rescoring")
        # Distinct RNG per (method, original restart index), independent
        # of task chunking, CPU process scheduling, or number of workers.
        method_code = {"KL": 101, "Tabu": 202}[method]
        rng = np.random.default_rng(np.random.SeedSequence([
            job["seed"], int(Path(job["graph_path"]).stem[1:]),
            int(restart), method_code,
        ]))
        if method == "KL":
            result, diagnostics = optimize_kl(
                _WORK_GRAPH, original, rng, passes=job["kl_passes"],
            )
        elif method == "Tabu":
            result, diagnostics = optimize_tabu(
                _WORK_GRAPH, original, rng,
                steps=job["tabu_steps"], tenure=job["tabu_tenure"],
                stagnation=job["tabu_stagnation"],
            )
        else:
            raise ValueError("Invalid search method")
        score = float(cut_weight(_WORK_EDGES, result))
        if score + 1e-8 < initial or not np.isclose(
            score, diagnostics["cut_weight"], atol=1e-8, rtol=0,
        ):
            raise AssertionError("Refined score invalid or worse than start")
        refined.append(result)
        scores.append(score)
        trials.append({
            "restart": int(restart),
            "initial_cut": initial,
            "final_cut": score,
            "improvement": score - initial,
            **diagnostics,
        })
    trial_path = Path(job["trial_path"])
    archive = new_archive(
        Path(job["output_dir"]) / "archives", new_run_id(),
        Path(job["graph_path"]), _WORK_GRAPH.shape[0], _WORK_EDGES,
        2, config, jsonl_output=trial_path, argv=[
            "experiment010", method, job["graph_path"],
        ],
    )
    save_method(archive, archive_method, refined, scores, k=2)
    update_manifest(archive, status="complete", solver_info={
        "algorithm": method,
        "original_1flip_local_optimum_sources": True,
        "initial_max": max(x["initial_cut"] for x in trials),
        "refined_max": max(scores),
        "source_method": config["source_method"],
        "no_second_eva_or_rounding": True,
    })
    check = verify_archive(archive["path"])
    if (check["status"] != "complete"
            or check["graph_sha256"] != config["graph_sha256"]
            or check["methods"][archive_method]["verified_partitions"]
            != len(refined)):
        raise RuntimeError("Refined archive did not independently verify")

    record = {
        "status": "complete",
        "graph": Path(job["graph_path"]).stem,
        "method": method,
        "source_method": config["source_method"],
        "factor_sha256": config["factor_sha256"],
        "graph_sha256": config["graph_sha256"],
        "settings": config,
        "archive_method": archive_method,
        "archive_path": str(archive["path"]),
        "trials_path": str(trial_path),
        "trials": trials,
        "best_initial_cut": max(x["initial_cut"] for x in trials),
        "best_final_cut": max(scores),
        "improvement_of_best": max(scores) - max(x["initial_cut"] for x in trials),
    }
    trial_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = trial_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf8") as out:
        for trial in trials:
            out.write(json.dumps({
                "graph": record["graph"], "method": method,
                "source_method": config["source_method"],
                "graph_sha256": config["graph_sha256"],
                "factor_sha256": config["factor_sha256"],
                "archive_path": record["archive_path"],
                **trial,
            }, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, trial_path)
    _write_json(case_path, record)
    return record, False


def write_summary(output_dir, *, benchmark=None):
    output_dir = Path(output_dir)
    fields = (
        "graph", "source_method", "method", "restart",
        "initial_cut", "final_cut", "improvement", "seconds",
        "termination", "passes_improved", "flips_retained",
        "steps", "nonpositive_moves", "new_best_events",
        "aspiration_moves", "tenure", "stagnation",
        "graph_sha256", "factor_sha256", "archive_path",
        "delta_vs_benchmark",
    )
    result_path = output_dir / "summary.csv"
    temp = result_path.with_suffix(".tmp")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with temp.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for path in sorted((output_dir / "cases").glob("G*.json")):
            record = json.loads(path.read_text(encoding="utf8"))
            for trial in record["trials"]:
                writer.writerow({
                    **{k: trial.get(k) for k in fields},
                    "graph": record["graph"],
                    "source_method": record["source_method"],
                    "method": record["method"],
                    "graph_sha256": record["graph_sha256"],
                    "factor_sha256": record["factor_sha256"],
                    "archive_path": record["archive_path"],
                    "delta_vs_benchmark": (
                        float(trial["final_cut"]) - benchmark
                        if benchmark is not None else None
                    ),
                })
    os.replace(temp, result_path)
    return result_path


def run_graph(
    path, *, source_dir, output_dir, methods=METHODS,
    source_method="K-MeansNM-Flip_EVA", top_starts=200, chunk_size=20,
    workers=2, seed=2026, kl_passes=30, tabu_steps=40000,
    tabu_tenure=25, tabu_stagnation=8000, factor_sha_prefix=None,
    benchmark=None,
):
    path, source_dir, output_dir = (
        Path(path), Path(source_dir), Path(output_dir)
    )
    if (top_starts < 0 or chunk_size < 1 or workers < 1 or seed < 0
            or kl_passes < 1 or tabu_steps < 1
            or tabu_tenure < 1 or tabu_stagnation < 1):
        raise ValueError("Invalid restart counts, worker count or search budget")
    if len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("Expected nonduplicated KL and/or Tabu method names")
    graph_sha = _file_sha256(path)
    factor_sha, sources, provenance = _read_initials(
        path, source_dir, source_method, factor_sha_prefix,
    )
    ids = _select_starts(sources, top_starts)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_max = float(max(score for _, score in sources.values()))
    best_selected = float(max(sources[i][1] for i in ids))
    print(f"{path.stem}: verified {len(sources)} archived "
          f"{source_method} starts, using top {len(ids)}; "
          f"all-source best={source_max:g}, selected best={best_selected:g}",
          flush=True)

    jobs = []
    for start in range(0, len(ids), chunk_size):
        sample = ids[start:start + chunk_size]
        chunk = [
            {"restart": int(i), "initial_cut": float(sources[i][1]),
             "labels": sources[i][0]} for i in sample
        ]
        for method in methods:
            token, config = _case_identity(
                method, chunk, graph_sha, factor_sha, source_method,
                seed, kl_passes, tabu_steps, tabu_tenure,
                tabu_stagnation, provenance,
            )
            prefix = f"{path.stem}-{source_method}-{method}-{token}"
            jobs.append({
                "method": method, "settings": config,
                "graph_path": str(path), "starts": chunk,
                "case_path": str(output_dir / "cases" / f"{prefix}.json"),
                "trial_path": str(output_dir / "trials" / f"{prefix}.jsonl"),
                "output_dir": str(output_dir), "seed": seed,
                "kl_passes": kl_passes, "tabu_steps": tabu_steps,
                "tabu_tenure": tabu_tenure,
                "tabu_stagnation": tabu_stagnation,
            })
    outcomes = []
    errors = []
    # Schedule interleaved KL/Tabu chunks so both methods start
    # concurrently when --workers >= 2. Uses Windows spawn-safe
    # top-level functions and no shared writable files per chunk.
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker,
        initargs=(str(path),),
    ) as pool:
        futures = {pool.submit(_run_chunk, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                record, skipped = future.result()
                outcomes.append(record)
                print(
                    f"{record['graph']} {record['method']} "
                    f"n={len(record['trials'])} "
                    f"{'verified skip' if skipped else 'complete'}: "
                    f"best {record['best_initial_cut']:g} -> "
                    f"{record['best_final_cut']:g} "
                    f"(+{record['improvement_of_best']:g})",
                    flush=True,
                )
            except Exception as exc:
                errors.append({
                    "graph": path.stem, "method": job["method"],
                    "case_path": job["case_path"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc().replace(
                        str(Path.home()), "<HOME>"
                    ),
                })
                _write_json(
                    output_dir / "errors" / (
                        Path(job["case_path"]).stem + ".json"
                    ), errors[-1],
                )
                print(f"{path.stem} {job['method']}: FAILED {exc}", flush=True)
            write_summary(output_dir, benchmark=benchmark)
    if errors:
        raise RuntimeError(
            f"{len(errors)} chunk(s) failed; diagnostic files in "
            f"{output_dir / 'errors'}"
        )
    statistics = {}
    for method in methods:
        trials = [
            trial for case in outcomes if case["method"] == method
            for trial in case["trials"]
        ]
        scores = np.asarray([row["final_cut"] for row in trials])
        initial = np.asarray([row["initial_cut"] for row in trials])
        statistics[method] = {
            "count": len(trials),
            "best_cut": float(scores.max()),
            "mean_cut": float(scores.mean()),
            "median_cut": float(np.median(scores)),
            "sample_std_cut": float(scores.std(ddof=1))
            if len(scores) > 1 else None,
            "best_restart": int(trials[int(np.argmax(scores))]["restart"]),
            "best_start": float(initial.max()),
            "mean_improvement": float((scores-initial).mean()),
            "improved_fraction": float(np.mean(scores > initial + 1e-8)),
            "seconds_sum": float(sum(t["seconds"] for t in trials)),
        }
    report = {
        "graph": path.stem,
        "source_method": source_method,
        "source_count": len(sources),
        "selected_restart_ids": ids,
        "source_best_cut": source_max,
        "best_selected_start": best_selected,
        "graph_sha256": graph_sha,
        "factor_sha256": factor_sha,
        "source_segments": provenance,
        "settings": {
            "top_starts": top_starts,
            "chunk_size": chunk_size, "workers": workers,
            "seed": seed, "kl_passes": kl_passes,
            "tabu_steps": tabu_steps, "tabu_tenure": tabu_tenure,
            "tabu_stagnation": tabu_stagnation,
        },
        "methods": statistics,
        "benchmark": benchmark,
    }
    # Preserve summaries from the top-200 pilot when later extending to
    # all 1,000 starts or changing budgets. Completed archives are append-only.
    summary_signature = hashlib.sha256(json.dumps({
        "graph": path.stem, "factor": factor_sha,
        "source_method": source_method, "selected_ids": ids,
        "methods": list(methods), "settings": report["settings"],
    }, sort_keys=True).encode("utf8")).hexdigest()[:16]
    _write_json(
        output_dir / "summaries" /
        f"{path.stem}-{source_method}-{factor_sha[:16]}-{summary_signature}.json",
        report,
    )
    print(
        f"{path.stem}: " +
        "; ".join(
            f"{name} best={data['best_cut']:g} "
            f"(mean={data['mean_cut']:.2f}, "
            f"+{data['best_cut']-best_selected:g} vs best input)"
            for name, data in statistics.items()
        ), flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=["G63"])
    parser.add_argument("--gset-dir", type=Path, default=Path("data"))
    parser.add_argument("--source-dir", type=Path, default=Path(
        "results/gset/experiment008-eva-rounding-flips",
    ))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "results/gset/experiment010-eva-kl-tabu",
    ))
    parser.add_argument(
        "--source-method", choices=(
            "K-MeansNM-Flip_EVA", "FJ-Flip_EVA",
        ), default="K-MeansNM-Flip_EVA",
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS))
    parser.add_argument("--top-starts", type=int, default=200,
                        help="Best archived K-means starts to refine; 0 = ALL")
    parser.add_argument("--chunk-size", type=int, default=20,
                        help="Checkpoint/archive size in starts per method")
    parser.add_argument("--workers", type=int, default=2,
                        help="Parallel independent solver processes")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--kl-passes", type=int, default=30)
    parser.add_argument("--tabu-steps", type=int, default=40000)
    parser.add_argument("--tabu-tenure", type=int, default=25)
    parser.add_argument("--tabu-stagnation", type=int, default=8000)
    parser.add_argument("--factor-sha-prefix")
    parser.add_argument("--benchmark", type=float)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--push-branch")
    args = parser.parse_args()
    if args.push and not args.push_branch:
        parser.error("--push requires --push-branch")
    if args.output_dir.resolve() == args.source_dir.resolve():
        parser.error("Output directory must differ from source segment directory")
    if (args.workers < 1 or args.chunk_size < 1
            or args.top_starts < 0 or args.kl_passes < 1
            or args.tabu_steps < 1 or args.tabu_tenure < 1
            or args.tabu_stagnation < 1 or args.seed < 0):
        parser.error("Invalid search budget or settings")
    paths = discover_graphs(args.gset_dir, args.graphs)
    failures = 0
    for path in paths:
        try:
            run_graph(
                path, source_dir=args.source_dir,
                output_dir=args.output_dir, methods=args.methods,
                source_method=args.source_method,
                top_starts=args.top_starts, chunk_size=args.chunk_size,
                workers=args.workers, seed=args.seed,
                kl_passes=args.kl_passes, tabu_steps=args.tabu_steps,
                tabu_tenure=args.tabu_tenure,
                tabu_stagnation=args.tabu_stagnation,
                factor_sha_prefix=args.factor_sha_prefix,
                benchmark=args.benchmark,
            )
            (args.output_dir / "errors" / f"{path.stem}.json").unlink(
                missing_ok=True
            )
        except (OSError, ValueError, RuntimeError, AssertionError,
                ArithmeticError, KeyError) as exc:
            failures += 1
            _write_json(
                args.output_dir / "errors" / f"{path.stem}.json", {
                    "graph": path.stem, "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc().replace(
                        str(Path.home()), "<HOME>"
                    ),
                },
            )
            print(f"{path.stem}: ERROR {exc}", flush=True)
            if not args.continue_on_error:
                if args.push:
                    _git_push(
                        args.output_dir, "experiment010 partial KL/Tabu results",
                        allowed_branch=args.push_branch,
                    )
                raise
    if args.push:
        _git_push(
            args.output_dir, "experiment010 KL/Tabu signed Max-Cut results",
            allowed_branch=args.push_branch,
        )
    print(
        f"Experiment010: {len(paths)} graphs, errors={failures}; "
        f"results={args.output_dir / 'summary.csv'}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
