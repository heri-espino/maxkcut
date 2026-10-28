"""Experiment012: optimize K-MeansNM ROUNDING first, THEN generalized KL.

Inputs: independently verified archived experiment008 K-MeansNM-Flip
partitions and the EXACT original-space EVA factor. For every saved
restart, reconstruct its K-MeansNM seeded direction trajectory, optimize
the original signed-weight Max-Cut threshold exactly for each visited
direction, explore perturbed/guided directions by the same objective,
then optimize the resulting binary labels with accelerated KL.

As a paired CONTROL, run KL directly on the original archived
K-MeansNM-Flip partition, reproducing experiment010's KL per-restart
seed schedule when using --seed 2026 and --kl-passes 30. This isolates
new rounding from subsequent local optimization.

Four outputs are independently archived for every restart:
  original saved K-MeansNM-Flip baseline (input reference)
  cut-aware threshold/direction result BEFORE KL
  original K-MeansNM-Flip + KL (paired control)
  cut-aware K-MeansNM + KL (proposed method)

The original graph signed weighted Max-Cut is used in every comparison.
No new EVA solve, PCA, MST, or SDPs. Windows-safe concurrent workers
checkpoint immutable batches; identical reruns verify and skip them.
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
    _file_sha256, load_method, new_archive, new_run_id,
    save_method, update_manifest, verify_archive,
)
from .eva_cli import _git_push, _write_json
from .eva_cutaware_kmeans import improve_kmeans_direction
from .eva_elite_search import optimize_kl_fast
from .eva_escape_cli import _read_initials, _select_starts
from .eva_rounding_cli import _factor_and_case, _immutable_factor_copy
from .io import edge_arrays
from .objective import cut_weight
from .sweep import discover_graphs


_GRAPH = None
_EDGES = None
_Y = None
_GRAPH_SHA = None
_FACTOR_SHA = None

METHODS = (
    "KMeansNMFlipInput",
    "CutAwarePreKL",
    "KMeansNMFlipKL",
    "CutAwareKMeansKL",
)


def _init_worker(graph_path, factor_dir, expected_graph_sha, expected_factor_sha):
    global _GRAPH, _EDGES, _Y, _GRAPH_SHA, _FACTOR_SHA
    graph, y, _, graph_sha, factor_sha = _factor_and_case(
        Path(graph_path), Path(factor_dir),
    )
    if graph_sha != expected_graph_sha or factor_sha != expected_factor_sha:
        raise ValueError("Saved EVA factor/graph changed during experiment012")
    _GRAPH, _EDGES, _Y = graph, edge_arrays(graph), y
    _GRAPH_SHA, _FACTOR_SHA = graph_sha, factor_sha


def _config(chunk, *, graph_sha, factor_sha, source_npzs,
            seed, rounding_seed, direction_trials, kmeans_iters,
            perturb_scales, kl_passes, target):
    settings = {
        "version": "experiment012-kmeans-threshold-direction-kl-v1",
        "graph_sha256": graph_sha,
        "factor_sha256": factor_sha,
        "input_source_method": "K-MeansNM-Flip_EVA",
        "source_npz_sha256": sorted(source_npzs),
        "restarts": [int(x["restart"]) for x in chunk],
        "input_cuts": [float(x["initial_cut"]) for x in chunk],
        "seed": seed,
        "rounding_seed": rounding_seed,
        "direction_trials": direction_trials,
        "kmeans_iters": kmeans_iters,
        "perturb_scales": list(perturb_scales),
        "kl_passes": kl_passes,
        "target": target,
        "objective": "original_signed_weighted_max_cut",
    }
    token = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf8"),
    ).hexdigest()[:20]
    return token, settings


def _verify_saved(record, settings):
    if record.get("status") != "complete" or record.get("settings") != settings:
        raise ValueError("Saved cut-aware K-Means settings/source mismatch")
    checked = verify_archive(record["archive_path"])
    if checked["status"] != "complete" or checked["graph_sha256"] != settings["graph_sha256"]:
        raise RuntimeError("Archived cut-aware partition verification failed")
    archive_path = Path(record["archive_path"])
    if len(record["trials"]) != len(settings["restarts"]):
        raise RuntimeError("Saved cut-aware restart count mismatch")
    for name in METHODS:
        archive_method = f"Exp012-{name}_EVA"
        if checked["methods"][archive_method]["verified_partitions"] != len(record["trials"]):
            raise RuntimeError("Missing cut-aware stage in partition archive")
        _, scores = load_method(archive_path / f"{archive_method}.npz")
        if not np.allclose(
            scores, [item["scores"][name] for item in record["trials"]],
            atol=1e-8, rtol=0,
        ):
            raise RuntimeError("Cut-aware archive contains mismatched weighted cuts")
    for trial, expected_restart, initial in zip(
        record["trials"], settings["restarts"], settings["input_cuts"],
    ):
        if trial["restart"] != expected_restart or trial["scores"]["KMeansNMFlipInput"] != initial:
            raise RuntimeError("Cut-aware source restart identity mismatch")
        if trial["scores"]["KMeansNMFlipKL"] + 1e-8 < initial:
            raise RuntimeError("Paired control KL deteriorated original cut")
        if trial["scores"]["CutAwareKMeansKL"] + 1e-8 < trial["scores"]["CutAwarePreKL"]:
            raise RuntimeError("Cut-aware KL worsened its starting partition")
    return record


def _solve_chunk(job):
    case_path = Path(job["case_path"])
    config = job["settings"]
    if case_path.is_file():
        return _verify_saved(
            json.loads(case_path.read_text(encoding="utf8")), config,
        ), True
    if _GRAPH is None or _Y is None:
        raise RuntimeError("Worker graph/EVA factor was not initialized")
    if _GRAPH_SHA != config["graph_sha256"] or _FACTOR_SHA != config["factor_sha256"]:
        raise RuntimeError("Worker factor/graph identity mismatch")

    partitions = {name: [] for name in METHODS}
    score_lists = {name: [] for name in METHODS}
    trials = []
    for item in job["starts"]:
        restart = int(item["restart"])
        baseline = np.asarray(item["labels"], dtype=np.int8)
        baseline_score = float(cut_weight(_EDGES, baseline))
        if not np.isclose(
            baseline_score, item["initial_cut"], rtol=0, atol=1e-8,
        ):
            raise RuntimeError(f"Source restart {restart} fails independent cut verification")

        # The K-MeansNM-Flip rounding was originally seeded with
        # SeedSequence([rounding_seed, graph_number, 2, restart, 0]).
        directions_rng = np.random.default_rng(np.random.SeedSequence([
            job["rounding_seed"], job["graph_number"], 2, restart, 0,
        ]))
        # Reuse the exact original K-MeansNM initialization and Lloyd
        # trajectory, then extend its search with signed-cut-aware
        # threshold optimization and seeded direction perturbations.
        aware_labels, aware_report = improve_kmeans_direction(
            _Y, _GRAPH, directions_rng,
            direction_trials=job["direction_trials"],
            kmeans_iters=job["kmeans_iters"],
            perturb_scales=job["perturb_scales"],
            target=job["target"],
        )

        # Reproduce experiment010's deterministic KL control schedule.
        control_rng = np.random.default_rng(np.random.SeedSequence([
            job["seed"], job["graph_number"], restart, 101,
        ]))
        control_labels, control_report = optimize_kl_fast(
            _GRAPH, baseline.copy(), control_rng,
            passes=job["kl_passes"],
        )
        aware_kl_rng = np.random.default_rng(np.random.SeedSequence([
            job["seed"], job["graph_number"], restart, 12013,
        ]))
        aware_kl_labels, aware_kl_report = optimize_kl_fast(
            _GRAPH, aware_labels.copy(), aware_kl_rng,
            passes=job["kl_passes"],
        )
        scores = {
            "KMeansNMFlipInput": baseline_score,
            "CutAwarePreKL": float(cut_weight(_EDGES, aware_labels)),
            "KMeansNMFlipKL": float(cut_weight(_EDGES, control_labels)),
            "CutAwareKMeansKL": float(cut_weight(_EDGES, aware_kl_labels)),
        }
        if (scores["CutAwareKMeansKL"] + 1e-8 < scores["CutAwarePreKL"]
                or scores["KMeansNMFlipKL"] + 1e-8 < baseline_score
                or not np.isclose(scores["CutAwarePreKL"], aware_report["cut"], atol=1e-8, rtol=0)):
            raise RuntimeError("Cut-aware stage or KL violated monotone refinement")
        for name, part in (
            ("KMeansNMFlipInput", baseline),
            ("CutAwarePreKL", aware_labels),
            ("KMeansNMFlipKL", control_labels),
            ("CutAwareKMeansKL", aware_kl_labels),
        ):
            partitions[name].append(np.asarray(part, dtype=np.int8))
            score_lists[name].append(float(scores[name]))
        trials.append({
            "restart": restart,
            "scores": scores,
            "kmeans_raw_cut": aware_report["raw_kmeans_best_cut"],
            "kmeans_iterations": aware_report["kmeans_iterations"],
            "direction_candidates": aware_report["direction_candidates"],
            "direction_improvements": aware_report["improvement_trace"],
            "best_direction_source": aware_report["best_source"],
            "best_threshold": aware_report["best_threshold"],
            "direction_seconds": aware_report["seconds"],
            "control_kl_seconds": control_report["seconds"],
            "aware_kl_seconds": aware_kl_report["seconds"],
            "control_kl_passes_improved": control_report["passes_improved"],
            "aware_kl_passes_improved": aware_kl_report["passes_improved"],
            "max_score": max(scores.values()),
            "target_reached": bool(max(scores.values()) >= job["target"]),
        })

    trial_path = Path(job["trial_path"])
    archive = new_archive(
        Path(job["dest"]) / "archives", new_run_id(),
        Path(job["graph_path"]), _GRAPH.shape[0], _EDGES,
        2, config, jsonl_output=trial_path,
        argv=["experiment012", str(job["graph_path"])],
    )
    for name in METHODS:
        save_method(
            archive, f"Exp012-{name}_EVA",
            partitions[name], score_lists[name], k=2,
        )
    update_manifest(archive, status="complete", solver_info={
        "stage1": "Full-space antipodal spherical K-MeansNM",
        "stage2": "Exact signed Max-Cut threshold plus sampled direction improvements",
        "stage3": "Generalized KL, Numba heap",
        "no_eva_rerun": True,
        "no_pca": True,
        "best_by_stage": {
            name: float(max(score_lists[name])) for name in METHODS
        },
    })
    verified = verify_archive(archive["path"])
    if (verified["status"] != "complete"
            or verified["graph_sha256"] != config["graph_sha256"]
            or any(verified["methods"][f"Exp012-{name}_EVA"]["verified_partitions"]
                   != len(trials) for name in METHODS)):
        raise RuntimeError("New cut-aware K-Means archive verification failed")
    record = {
        "status": "complete", "graph": Path(job["graph_path"]).stem,
        "factor_sha256": config["factor_sha256"],
        "graph_sha256": config["graph_sha256"],
        "settings": config,
        "archive_path": str(archive["path"]),
        "trial_path": str(trial_path),
        "trials": trials,
        "best_by_stage": {
            name: float(max(score_lists[name])) for name in METHODS
        },
    }
    trial_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = trial_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf8") as stream:
        for trial in trials:
            stream.write(json.dumps({
                "graph": record["graph"],
                "graph_sha256": config["graph_sha256"],
                "factor_sha256": config["factor_sha256"],
                "archive_path": record["archive_path"],
                **trial,
            }, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, trial_path)
    _write_json(case_path, record)
    return record, False


def write_summary(dest, *, target):
    dest = Path(dest)
    output = dest / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "graph", "restart", "stage", "cut", "improvement_vs_input",
        "gap_to_target", "direction_candidates", "direction_seconds",
        "control_kl_seconds", "aware_kl_seconds", "archive_path",
    )
    tmp = output.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for path in sorted((dest / "cases").glob("G*.json")):
            case = json.loads(path.read_text(encoding="utf8"))
            for trial in case["trials"]:
                for stage in METHODS:
                    value = float(trial["scores"][stage])
                    writer.writerow({
                        "graph": case["graph"],
                        "restart": trial["restart"], "stage": stage,
                        "cut": value,
                        "improvement_vs_input": value - trial["scores"]["KMeansNMFlipInput"],
                        "gap_to_target": target-value,
                        "direction_candidates": trial["direction_candidates"],
                        "direction_seconds": trial["direction_seconds"],
                        "control_kl_seconds": trial["control_kl_seconds"],
                        "aware_kl_seconds": trial["aware_kl_seconds"],
                        "archive_path": case["archive_path"],
                    })
    os.replace(tmp, output)
    return output


def run_graph(
    path, *, factor_dir, rounding_dir, dest,
    top_starts=200, chunk_size=10, workers=4,
    seed=2026, direction_trials=96, kmeans_iters=100,
    perturb_scales=(0.03,0.08,0.16,0.35), kl_passes=30,
    target=27047., factor_sha_prefix=None,
):
    path, factor_dir, rounding_dir, dest = (
        Path(path), Path(factor_dir), Path(rounding_dir), Path(dest)
    )
    if (top_starts < 0 or chunk_size < 1 or workers < 1 or seed < 0
            or direction_trials < 0 or kmeans_iters < 1
            or kl_passes < 1 or target < 0):
        raise ValueError("Invalid cut-aware K-Means search parameters")
    graph, y, eva_case, graph_sha, factor_sha = _factor_and_case(
        path, factor_dir,
    )
    if factor_sha_prefix and not factor_sha.startswith(factor_sha_prefix):
        raise ValueError("Requested factor SHA prefix does not match source factor")
    src_factor, source, provenance = _read_initials(
        path, rounding_dir, "K-MeansNM-Flip_EVA",
        factor_sha_prefix,
    )
    if factor_sha != src_factor:
        raise ValueError("EVA factor is not the same as archived K-MeansNM-Flip source factor")
    seeds = {r["rounding_seed"] for r in provenance}
    if len(seeds) != 1:
        raise ValueError("Saved K-MeansNM-Flip segments have conflicting rounding seeds")
    rounding_seed = int(next(iter(seeds)))
    selected_ids = _select_starts(source, top_starts)
    _immutable_factor_copy(factor_dir, dest, path.stem, factor_sha)
    npzs = sorted(r["partition_npz_sha256"] for r in provenance)
    jobs = []
    for first in range(0, len(selected_ids), chunk_size):
        chunk = [
            {"restart": int(idx), "labels": source[idx][0],
             "initial_cut": float(source[idx][1])}
            for idx in selected_ids[first:first+chunk_size]
        ]
        token, settings = _config(
            chunk, graph_sha=graph_sha, factor_sha=factor_sha,
            source_npzs=npzs, seed=seed, rounding_seed=rounding_seed,
            direction_trials=direction_trials, kmeans_iters=kmeans_iters,
            perturb_scales=perturb_scales, kl_passes=kl_passes,
            target=target,
        )
        name = f"{path.stem}-CutAware-{token}"
        jobs.append({
            "graph_path": str(path),
            "case_path": str(dest / "cases" / f"{name}.json"),
            "trial_path": str(dest / "trials" / f"{name}.jsonl"),
            "dest": str(dest), "starts": chunk, "settings": settings,
            "rounding_seed": rounding_seed,
            "seed": seed, "graph_number": int(path.stem[1:]),
            "direction_trials": direction_trials,
            "kmeans_iters": kmeans_iters,
            "perturb_scales": tuple(perturb_scales),
            "kl_passes": kl_passes, "target": target,
        })
    print(
        f"{path.stem}: verified {len(source)} saved K-MeansNM-Flip starts "
        f"for EVA factor R^{y.shape[1]}; selected top {len(selected_ids)}. "
        f"Round+KL pairs={len(selected_ids)}, directions per restart "
        f"up to {direction_trials} extra; workers={workers}; "
        f"target={target:g}.",
        flush=True,
    )
    records, errors = [], []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker,
        initargs=(str(path), str(factor_dir), graph_sha, factor_sha),
    ) as pool:
        futures = {pool.submit(_solve_chunk, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                record, skipped = future.result()
                records.append(record)
                print(
                    f"{path.stem} cut-aware "
                    f"{'verified skip' if skipped else 'complete'} "
                    f"n={len(record['trials'])}: "
                    f"before KL={record['best_by_stage']['CutAwarePreKL']:g}, "
                    f"after KL={record['best_by_stage']['CutAwareKMeansKL']:g}, "
                    f"control KL={record['best_by_stage']['KMeansNMFlipKL']:g}, "
                    f"target gap={target-max(record['best_by_stage'].values()):+g}",
                    flush=True,
                )
            except Exception as exc:
                error = {
                    "graph": path.stem,
                    "case_path": job["case_path"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc().replace(
                        str(Path.home()), "<HOME>"
                    ),
                }
                errors.append(error)
                _write_json(
                    dest / "errors" / f"{Path(job['case_path']).stem}.json",
                    error,
                )
                print(f"{path.stem} cut-aware FAILED: {exc}", flush=True)
            write_summary(dest, target=target)
    if errors:
        raise RuntimeError(f"{len(errors)} chunks failed; see {dest/'errors'}")

    summary = {
        "graph": path.stem,
        "graph_sha256": graph_sha, "factor_sha256": factor_sha,
        "factor_rank": y.shape[1],
        "rounding_seed": rounding_seed,
        "selected_count": len(selected_ids),
        "selected_restarts": selected_ids,
        "source_count": len(source),
        "target": target,
        "settings": {
            "seed": seed, "direction_trials": direction_trials,
            "kmeans_iters": kmeans_iters,
            "perturb_scales": list(perturb_scales),
            "kl_passes": kl_passes, "workers": workers,
            "chunk_size": chunk_size,
        },
        "stages": {},
    }
    for stage in METHODS:
        values = np.asarray([
            row["scores"][stage] for record in records
            for row in record["trials"]
        ], dtype=float)
        summary["stages"][stage] = {
            "count": len(values),
            "best": float(values.max()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std(ddof=1)) if len(values)>1 else None,
            "gap_to_target": float(target-values.max()),
            "fraction_beating_target": float(np.mean(values>=target)),
        }
    fingerprint = hashlib.sha256(json.dumps({
        "stage": "experiment012",
        "graph_sha": graph_sha, "factor_sha": factor_sha,
        "selected_restarts": selected_ids,
        "settings": summary["settings"],
    }, sort_keys=True).encode("utf8")).hexdigest()[:16]
    _write_json(
        dest / "summaries" / f"{path.stem}-{fingerprint}.json",
        summary,
    )
    print(
        f"{path.stem}: " + "; ".join(
            f"{method} best={value['best']:g} "
            f"(target gap {value['gap_to_target']:+g})"
            for method, value in summary["stages"].items()
        ), flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=["G63"])
    parser.add_argument("--gset-dir", type=Path, default=Path("data"))
    parser.add_argument("--factor-dir", type=Path, default=Path(
        "results/gset/experiment003-eva-numba",
    ))
    parser.add_argument("--rounding-dir", type=Path, default=Path(
        "results/gset/experiment008-eva-rounding-flips",
    ))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "results/gset/experiment012-cut-aware-kmeans-kl",
    ))
    parser.add_argument("--top-starts", type=int, default=200,
                        help="Best original rounded partitions (0 = ALL)")
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026,
                        help="KL/control/direction extension seed")
    parser.add_argument("--direction-trials", type=int, default=96)
    parser.add_argument("--kmeans-iters", type=int, default=100)
    parser.add_argument("--perturb-scales", nargs="+", type=float,
                        default=[0.03,0.08,0.16,0.35])
    parser.add_argument("--kl-passes", type=int, default=30)
    parser.add_argument("--factor-sha-prefix")
    parser.add_argument("--target", type=float, default=27047.)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--push-branch")
    args = parser.parse_args()
    if args.push and not args.push_branch:
        parser.error("--push requires --push-branch")
    if (args.output_dir.resolve() == args.rounding_dir.resolve()
            or args.output_dir.resolve() == args.factor_dir.resolve()):
        parser.error("Output directory must be distinct from source directories")
    paths = discover_graphs(args.gset_dir, args.graphs)
    failures = 0
    for path in paths:
        try:
            run_graph(
                path, factor_dir=args.factor_dir,
                rounding_dir=args.rounding_dir, dest=args.output_dir,
                top_starts=args.top_starts, chunk_size=args.chunk_size,
                workers=args.workers, seed=args.seed,
                direction_trials=args.direction_trials,
                kmeans_iters=args.kmeans_iters,
                perturb_scales=tuple(args.perturb_scales),
                kl_passes=args.kl_passes,
                target=args.target,
                factor_sha_prefix=args.factor_sha_prefix,
            )
            (args.output_dir / "errors" / f"{path.stem}.json").unlink(
                missing_ok=True,
            )
        except (OSError, ValueError, KeyError, RuntimeError,
                AssertionError, ArithmeticError) as exc:
            failures += 1
            _write_json(
                args.output_dir / "errors" / f"{path.stem}.json", {
                    "graph": path.stem, "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc().replace(
                        str(Path.home()), "<HOME>"
                    ),
                },
            )
            print(f"{path.stem}: FAILED {exc}", flush=True)
            if not args.continue_on_error:
                if args.push:
                    _git_push(
                        args.output_dir, "experiment012 partial K-Means-KL",
                        allowed_branch=args.push_branch,
                    )
                raise
    if args.push:
        _git_push(
            args.output_dir, "experiment012 objective-aware K-Means-KL",
            allowed_branch=args.push_branch,
        )
    print(
        f"Experiment012: graphs={len(paths)}, errors={failures}; "
        f"summary={args.output_dir / 'summary.csv'}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
