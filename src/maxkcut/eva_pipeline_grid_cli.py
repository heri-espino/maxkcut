"""Experiment013 CLI: checkpointed, staged, parallel EVA Max-Cut pipeline grid.

The grid is a JSON file of method families and Cartesian hyperparameter
axes; see configs/experiment013-grid.json. All methods reuse the ONE
already completed EVA factor and original signed weighted graph.
Paired restart IDs are selected from independently verified experiment008
K-MeansNM-Flip archives, not new EVA solves.

For every (pipeline configuration, source restart) independently
archive the four stages: Round, Flip, KL, Final. A completed chunk is
verified and skipped; a larger top-N selection with identical settings
reuses the first chunks. Results are ranked by the exact signed graph
Max-Cut score; never claim that score is an upper bound or optimum.
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
from .eva_pipeline_grid import STAGES, evaluate_pipeline, expand_grid
from .eva_rounding_cli import _factor_and_case, _immutable_factor_copy
from .eva_escape_cli import _read_initials, _select_starts
from .io import edge_arrays
from .sweep import discover_graphs


_GRAPH = None
_Y = None
_EDGES = None
_HASHES = None


def _worker_init(path, factor_dir, expected):
    global _GRAPH, _Y, _EDGES, _HASHES
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    graph, vectors, case, graph_sha, factor_sha = _factor_and_case(
        Path(path), Path(factor_dir),
    )
    if [graph_sha, factor_sha] != expected:
        raise ValueError("Worker EVA factor/graph identity changed")
    _GRAPH, _Y = graph, vectors
    _EDGES = edge_arrays(graph)
    _HASHES = [graph_sha, factor_sha]


def _job_identity(config, selected, *, graph_sha, factor_sha,
                  source_partition_sha, source_segment_sha,
                  seed, rounding_seed, target):
    settings = {
        "version": "experiment013-eva-grid-v1",
        "graph_sha256": graph_sha,
        "factor_sha256": factor_sha,
        "source_method": "K-MeansNM-Flip_EVA",
        "source_partition_npz_sha256": sorted(source_partition_sha),
        "source_segment_sha256": sorted(source_segment_sha),
        "source_restarts": [int(x["restart"]) for x in selected],
        "source_scores": [float(x["source_cut"]) for x in selected],
        "rounding_seed": int(rounding_seed),
        "search_seed": int(seed),
        "target": float(target),
        "configuration": config,
    }
    key = hashlib.sha256(json.dumps(
        settings, sort_keys=True, allow_nan=False,
    ).encode("utf8")).hexdigest()[:20]
    return key, settings


def _verify_completed(record, settings):
    if record.get("status") != "complete" or record.get("settings") != settings:
        raise ValueError("Cached grid chunk has incompatible settings")
    verified = verify_archive(record["archive_path"])
    if verified["status"] != "complete" or verified["graph_sha256"] != settings["graph_sha256"]:
        raise ValueError("Grid archive graph or status mismatch")
    if len(record["trials"]) != len(settings["source_restarts"]):
        raise ValueError("Grid trial count differs from saved starts")
    for stage in STAGES:
        method = f"Grid-{stage}_EVA"
        if verified["methods"][method]["verified_partitions"] != len(record["trials"]):
            raise ValueError("Missing saved grid stage partitions")
        _, saved_scores = load_method(
            Path(record["archive_path"]) / f"{method}.npz",
        )
        if not np.allclose(
            saved_scores, [x["scores"][stage] for x in record["trials"]],
            atol=1e-8, rtol=0,
        ):
            raise ValueError("Grid archived stage scores do not match")
    for row, index, source_score in zip(
        record["trials"], settings["source_restarts"], settings["source_scores"],
    ):
        if row["restart"] != index or row["source_cut"] != source_score:
            raise ValueError("Cached grid paired restart mismatch")
        score = row["scores"]
        if not (score["Round"] <= score["Flip"]+1e-8
                and score["Flip"] <= score["KL"]+1e-8
                and score["KL"] <= score["Final"]+1e-8):
            raise ValueError("Cached grid stage cut unexpectedly decreased")
    return record


def _solve(job):
    case_path = Path(job["case_path"])
    settings = job["settings"]
    if case_path.is_file():
        record = json.loads(case_path.read_text(encoding="utf8"))
        return _verify_completed(record, settings), True
    if _GRAPH is None or _Y is None or _HASHES != [
        settings["graph_sha256"], settings["factor_sha256"],
    ]:
        raise RuntimeError("Worker graph and EVA factor are unavailable")
    partitions = {key: [] for key in STAGES}
    scores_lists = {key: [] for key in STAGES}
    trials = []
    for entry in job["starts"]:
        stages, scores, details, timings = evaluate_pipeline(
            _GRAPH, _Y, saved_labels=entry["labels"],
            saved_score=entry["source_cut"], restart=entry["restart"],
            graph_number=job["graph_number"],
            rounding_seed=job["rounding_seed"],
            search_seed=job["search_seed"],
            config=job["configuration"],
            target=job["target"],
        )
        for stage in STAGES:
            partitions[stage].append(stages[stage])
            scores_lists[stage].append(float(scores[stage]))
        trials.append({
            "restart": int(entry["restart"]),
            "source_cut": float(entry["source_cut"]),
            "scores": {k: float(v) for k, v in scores.items()},
            "details": details, "timings": timings,
        })

    trial_path = Path(job["trial_path"])
    archive = new_archive(
        Path(job["output_dir"]) / "archives", new_run_id(),
        Path(job["graph_path"]), _GRAPH.shape[0], _EDGES,
        2, settings, jsonl_output=trial_path,
        argv=["experiment013", job["configuration"]["id"]],
    )
    for stage in STAGES:
        save_method(
            archive, f"Grid-{stage}_EVA", partitions[stage],
            scores_lists[stage], k=2,
        )
    update_manifest(archive, status="complete", solver_info={
        "algorithm": "configurable EVA rounding + flips + KL + optional ILS",
        "source_factor_reused": True,
        "original_weighted_graph": True,
        "config_id": job["configuration"]["id"],
        "best_stage_scores": {
            k: float(max(scores_lists[k])) for k in STAGES
        },
    })
    check = verify_archive(archive["path"])
    if (check["status"] != "complete"
            or check["graph_sha256"] != settings["graph_sha256"]
            or any(check["methods"][f"Grid-{stage}_EVA"]["verified_partitions"]
                   != len(trials) for stage in STAGES)):
        raise RuntimeError("Grid results failed independent archive verification")
    record = {
        "status": "complete", "graph": Path(job["graph_path"]).stem,
        "configuration_id": job["configuration"]["id"],
        "configuration": job["configuration"],
        "settings": settings,
        "archive_path": str(archive["path"]),
        "trials_path": str(trial_path),
        "trials": trials,
        "best_final": float(max(scores_lists["Final"])),
    }
    trial_path.parent.mkdir(parents=True, exist_ok=True)
    temp_trial = trial_path.with_suffix(".tmp")
    with temp_trial.open("w", encoding="utf8") as stream:
        for row in trials:
            stream.write(json.dumps({
                "graph": record["graph"],
                "configuration_id": record["configuration_id"],
                "graph_sha256": settings["graph_sha256"],
                "factor_sha256": settings["factor_sha256"],
                "archive_path": str(archive["path"]),
                **row,
            }, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp_trial, trial_path)
    _write_json(case_path, record)
    return record, False


def _write_trial_summary(dest, graph_name, *, target):
    output = Path(dest) / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "graph", "config_id", "family", "rounder", "restart", "source_cut",
        "round_cut", "flip_cut", "kl_cut", "final_cut",
        "gain_vs_source", "gap_to_target", "total_seconds", "archive_path",
    ]
    tmp = output.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf8") as stream:
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader()
        for file in sorted((Path(dest) / "cases").glob(f"{graph_name}-grid-*.json")):
            record = json.loads(file.read_text(encoding="utf8"))
            cfg = record["configuration"]
            for trial in record["trials"]:
                score = trial["scores"]
                w.writerow({
                    "graph": graph_name, "config_id": cfg["id"],
                    "family": cfg["name"], "rounder": cfg["rounder"],
                    "restart": trial["restart"], "source_cut": trial["source_cut"],
                    "round_cut": score["Round"], "flip_cut": score["Flip"],
                    "kl_cut": score["KL"], "final_cut": score["Final"],
                    "gain_vs_source": score["Final"]-trial["source_cut"],
                    "gap_to_target": target-score["Final"],
                    "total_seconds": trial["timings"]["total_seconds"],
                    "archive_path": record["archive_path"],
                })
    os.replace(tmp, output)
    return output


def _aggregate(records, configs, *, selected_ids, graph_sha,
               factor_sha, target, source_best, seed):
    by_config = {cfg["id"]: [] for cfg in configs}
    for record in records:
        by_config[record["configuration_id"]].extend(record["trials"])
    standings = []
    for cfg in configs:
        # Jobs finish in arbitrary order under ProcessPoolExecutor.
        # Sort paired trials before floating-point aggregation, argmax
        # tie-breaking, and summary serialization. Otherwise a verified
        # rerun can produce a different leaderboard from identical cuts.
        rows = sorted(
            by_config[cfg["id"]],
            key=lambda trial: int(trial["restart"]),
        )
        if len(rows) != len(selected_ids):
            raise RuntimeError("Incomplete grid configuration in current ranking")
        if sorted(x["restart"] for x in rows) != sorted(selected_ids):
            raise RuntimeError("Different restart sets used for grid comparison")
        final = np.asarray([x["scores"]["Final"] for x in rows])
        source = np.asarray([x["source_cut"] for x in rows])
        top_index = int(np.argmax(final))
        timings = np.asarray([x["timings"]["total_seconds"] for x in rows])
        standings.append({
            "configuration_id": cfg["id"],
            "family": cfg["name"], "rounder": cfg["rounder"],
            "configuration": cfg,
            "count": len(rows),
            "best_cut": float(final.max()),
            "best_restart": int(rows[top_index]["restart"]),
            "mean_cut": float(final.mean()),
            "median_cut": float(np.median(final)),
            "std_cut": float(final.std(ddof=1)) if len(final)>1 else None,
            "best_start": float(source.max()),
            "mean_paired_gain": float((final-source).mean()),
            "improved_fraction": float(np.mean(final > source + 1e-8)),
            "seconds_total": float(timings.sum()),
            "seconds_mean": float(timings.mean()),
            "gap_to_target": float(target-final.max()),
        })
    # Rank purely by original-graph Max-Cut statistics. Runtime remains
    # reported but is intentionally NOT a tie-breaker: small wall-clock
    # differences across fresh runs would otherwise change promotion
    # selections despite identical cuts and seeds.
    standings.sort(key=lambda x: (
        -x["best_cut"], -x["mean_cut"], x["configuration_id"],
    ))
    for position, entry in enumerate(standings, 1):
        entry["rank"] = position
    return {
        "version": 1,
        "graph_sha256": graph_sha, "factor_sha256": factor_sha,
        "seed": seed, "target": target,
        "source_best_cut": source_best,
        "selected_restart_ids": list(selected_ids),
        "configurations_total": len(configs),
        "leaderboard": standings,
    }


def _write_rankings(dest, name, aggregate):
    fingerprint = hashlib.sha256(
        json.dumps({
            "graph": aggregate["graph_sha256"],
            "factor": aggregate["factor_sha256"],
            "config_ids": sorted(x["configuration_id"] for x in aggregate["leaderboard"]),
            "selected": aggregate["selected_restart_ids"],
            "seed": aggregate["seed"],
            "target": aggregate["target"],
        }, sort_keys=True).encode("utf8")
    ).hexdigest()[:16]
    report = Path(dest) / "leaderboards" / f"{name}-{fingerprint}.json"
    _write_json(report, aggregate)
    csv_path = report.with_suffix(".csv")
    fields = [
        "rank", "configuration_id", "family", "rounder", "count",
        "best_cut", "best_restart", "mean_cut", "median_cut",
        "std_cut", "best_start", "mean_paired_gain", "improved_fraction",
        "seconds_total", "seconds_mean", "gap_to_target",
    ]
    with csv_path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in aggregate["leaderboard"]:
            writer.writerow({key: row.get(key) for key in fields})
    return report, csv_path


def run_graph(
    path, *, factor_dir, rounding_dir, output_dir, grid_path,
    top_starts=40, chunk_size=10, workers=4, seed=2026,
    target=27047., promote_from=None, promote_top=None,
    factor_sha_prefix=None,
):
    path = Path(path)
    factor_dir, rounding_dir, output_dir = map(
        Path, (factor_dir, rounding_dir, output_dir),
    )
    if (top_starts < 0 or chunk_size < 1 or workers < 1 or seed < 0
            or not np.isfinite(target) or not Path(grid_path).is_file()):
        raise ValueError("Invalid grid settings or missing JSON grid")
    grid_spec = json.loads(Path(grid_path).read_text(encoding="utf8"))
    all_configs = expand_grid(grid_spec)
    graph, y, eva_case, graph_sha, factor_sha = _factor_and_case(
        path, factor_dir,
    )
    if factor_sha_prefix and not factor_sha.startswith(factor_sha_prefix):
        raise ValueError("EVA factor does not match requested factor SHA prefix")
    source_factor, all_starts, provenance = _read_initials(
        path, rounding_dir, "K-MeansNM-Flip_EVA", factor_sha_prefix,
    )
    if source_factor != factor_sha:
        raise ValueError("Saved experiment008 starts use a different EVA factor")
    rounding_seeds = {item["rounding_seed"] for item in provenance}
    if len(rounding_seeds) != 1:
        raise ValueError("Saved rounding sources disagree on seed")
    rounding_seed = int(next(iter(rounding_seeds)))
    ids = _select_starts(all_starts, top_starts)
    source_best = float(max(x[1] for x in all_starts.values()))
    if promote_from is not None:
        prior = json.loads(Path(promote_from).read_text(encoding="utf8"))
        if (prior["graph_sha256"] != graph_sha
                or prior["factor_sha256"] != factor_sha
                or not prior["leaderboard"]):
            raise ValueError("Promotion source graph/factor/leaderboard mismatch")
        if promote_top is None or promote_top < 1:
            raise ValueError("--promote-top must be >=1 with --promote-from")
        allowed_ids = {
            r["configuration_id"]
            for r in prior["leaderboard"][:promote_top]
        }
        all_configs = [
            config for config in all_configs if config["id"] in allowed_ids
        ]
        if len(all_configs) != len(allowed_ids):
            raise ValueError("Promotion source configs not in the current grid file")
    if not all_configs:
        raise ValueError("No grid configurations selected")
    output_dir.mkdir(parents=True, exist_ok=True)
    _immutable_factor_copy(factor_dir, output_dir, path.stem, factor_sha)
    parts_sha = [r["partition_npz_sha256"] for r in provenance]
    segment_sha = [r["segment_sha256"] for r in provenance]
    jobs = []
    for cfg in all_configs:
        for offset in range(0, len(ids), chunk_size):
            chunk = [
                {"restart": int(i), "labels": all_starts[i][0],
                 "source_cut": float(all_starts[i][1])}
                for i in ids[offset:offset+chunk_size]
            ]
            token, settings = _job_identity(
                cfg, chunk, graph_sha=graph_sha, factor_sha=factor_sha,
                source_partition_sha=parts_sha, source_segment_sha=segment_sha,
                seed=seed, rounding_seed=rounding_seed,
                target=target,
            )
            name = f"{path.stem}-grid-{cfg['id']}-{token}"
            jobs.append({
                "configuration": cfg,
                "starts": chunk, "settings": settings,
                "graph_path": str(path),
                "graph_number": int(path.stem[1:]),
                "search_seed": seed, "rounding_seed": rounding_seed,
                "target": target, "output_dir": str(output_dir),
                "case_path": str(output_dir/"cases"/f"{name}.json"),
                "trial_path": str(output_dir/"trials"/f"{name}.jsonl"),
            })
    print(
        f"{path.stem}: verified EVA rank={y.shape[1]}, "
        f"{len(all_starts)} saved K-MeansNM-Flip starts; "
        f"{len(all_configs)} configurations x {len(ids)} selected restarts; "
        f"{len(jobs)} checkpointed jobs; {workers} processes; "
        f"target={target:g} (previous verified source best={source_best:g})",
        flush=True,
    )
    results, errors = [], []
    overall_best = float("-inf")
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(str(path), str(factor_dir), [graph_sha, factor_sha]),
    ) as executor:
        futures = {executor.submit(_solve, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                record, skipped = future.result()
                results.append(record)
                best = record["best_final"]
                overall_best = max(overall_best, best)
                print(
                    f"{path.stem} {job['configuration']['name']}/"
                    f"{job['configuration']['id'][:8]} "
                    f"{'verified skip' if skipped else 'complete'} "
                    f"n={len(record['trials'])} best={best:g}; "
                    f"global grid best={overall_best:g} "
                    f"(gap={target-overall_best:+g})",
                    flush=True,
                )
            except Exception as exc:
                errors.append({
                    "graph": path.stem,
                    "config_id": job["configuration"]["id"],
                    "case": job["case_path"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc().replace(
                        str(Path.home()), "<HOME>",
                    ),
                })
                _write_json(
                    output_dir/"errors"/f"{Path(job['case_path']).stem}.json",
                    errors[-1],
                )
                print(f"{path.stem} grid job FAILED: {exc}", flush=True)
    _write_trial_summary(output_dir, path.stem, target=target)
    if errors:
        raise RuntimeError(
            f"{len(errors)} grid jobs failed; diagnostics in {output_dir/'errors'}. "
            "Successfully completed chunks are preserved."
        )
    aggregate = _aggregate(
        results, all_configs, selected_ids=ids,
        graph_sha=graph_sha, factor_sha=factor_sha,
        target=target, source_best=source_best, seed=seed,
    )
    report, csv_path = _write_rankings(output_dir, path.stem, aggregate)
    print(
        f"{path.stem} grid COMPLETE: "
        f"best={aggregate['leaderboard'][0]['best_cut']:g}, "
        f"configuration={aggregate['leaderboard'][0]['family']}, "
        f"gap to {target:g}: {aggregate['leaderboard'][0]['gap_to_target']:+g}; "
        f"ranking={csv_path}",
        flush=True,
    )
    return aggregate, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=["G63"])
    parser.add_argument("--gset-dir", type=Path, default=Path("data"))
    parser.add_argument("--factor-dir", type=Path,
                        default=Path("results/gset/experiment003-eva-numba"))
    parser.add_argument("--rounding-dir", type=Path,
                        default=Path("results/gset/experiment008-eva-rounding-flips"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/gset/experiment013-eva-grid"))
    parser.add_argument("--grid", type=Path,
                        default=Path("configs/experiment013-grid.json"))
    parser.add_argument("--top-starts", type=int, default=40,
                        help="Top original starts; 0 means all 1000")
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--target", type=float, default=27047.)
    parser.add_argument("--factor-sha-prefix")
    parser.add_argument("--promote-from", type=Path,
                        help="Previous leaderboard JSON for staged top-k promotion")
    parser.add_argument("--promote-top", type=int,
                        help="Number of best configurations to promote")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--push-branch")
    args = parser.parse_args()
    if args.push and not args.push_branch:
        parser.error("--push requires --push-branch")
    if args.output_dir.resolve() in (
        args.factor_dir.resolve(), args.rounding_dir.resolve(),
    ):
        parser.error("Output must differ from EVA and archived rounding inputs")
    if bool(args.promote_from) != bool(args.promote_top):
        parser.error("--promote-from and --promote-top must be supplied together")
    paths = discover_graphs(args.gset_dir, args.graphs)
    failures = 0
    for path in paths:
        try:
            run_graph(
                path, factor_dir=args.factor_dir,
                rounding_dir=args.rounding_dir, output_dir=args.output_dir,
                grid_path=args.grid, top_starts=args.top_starts,
                chunk_size=args.chunk_size, workers=args.workers,
                seed=args.seed, target=args.target,
                promote_from=args.promote_from, promote_top=args.promote_top,
                factor_sha_prefix=args.factor_sha_prefix,
            )
            (args.output_dir/"errors"/f"{path.stem}.json").unlink(
                missing_ok=True,
            )
        except (OSError, ValueError, RuntimeError, AssertionError,
                ArithmeticError, KeyError) as exc:
            failures += 1
            _write_json(args.output_dir/"errors"/f"{path.stem}.json", {
                "graph": path.stem,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().replace(
                    str(Path.home()), "<HOME>",
                ),
            })
            print(f"{path.stem}: grid error {exc}", flush=True)
            if not args.continue_on_error:
                if args.push:
                    _git_push(
                        args.output_dir, "experiment013 partial EVA grid",
                        allowed_branch=args.push_branch,
                    )
                raise
    if args.push:
        _git_push(
            args.output_dir, "experiment013 EVA grid results",
            allowed_branch=args.push_branch,
        )
    print(f"Experiment013: {len(paths)} graphs, errors={failures}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
