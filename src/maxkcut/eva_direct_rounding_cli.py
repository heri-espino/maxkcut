"""Experiment015: optimize ONLY rounding normals against hard signed Max-Cut.

Compare exact great-semicircle line search, antithetic stochastic
gradient ascent on Gaussian-smoothed ACTUAL hard cuts, and signed-edge
minibatch logistic SGD, each with optional final exact line search.

The original EVA matrix is loaded read-only, checksum verified, and
never optimized. No K-Means, SDP, or new EVA run. Each method uses the
SAME raw GW hyperplane per saved restart index. Four stages per trial:
original raw GW, optimized raw GW, old raw GW + KL, new raw GW + KL.
All original-graph signed cut scores are independently checked.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np

from .archive import (
    _atomic_npz, _file_sha256, load_method,
    new_archive, new_run_id, save_method,
    update_manifest, verify_archive,
)
from .eva_cli import _git_push, _write_json
from .eva_direct_rounding import optimize_hyperplane
from .eva_rounding_sga import optimize_es_sga, optimize_soft_sga
from .eva_elite_search import optimize_kl_fast
from .eva_rounding_cli import _factor_and_case, _immutable_factor_copy
from .io import edge_arrays
from .objective import cut_weight
from .sweep import discover_graphs


_GRAPH = None
_EDGES = None
_Y = None
_SOURCE = None

METHODS = ("linesweep", "es-sga", "es-sga-sweep",
           "soft-sga", "soft-sga-sweep")
STAGES = ("GWRaw", "OptimizedRaw", "GWRawKL", "OptimizedRawKL")


def _worker_init(path, factor_dir, expected):
    global _GRAPH, _EDGES, _Y, _SOURCE
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    graph, y, case, graph_sha, factor_sha = _factor_and_case(
        Path(path), Path(factor_dir),
    )
    if [graph_sha, factor_sha] != expected:
        raise ValueError("Original graph/EVA factor SHA mismatch")
    _GRAPH, _Y, _EDGES = graph, y, edge_arrays(graph)
    _SOURCE = [graph_sha, factor_sha]


def _identity(method, start_indices, graph_sha, factor_sha, *, settings):
    config = {
        "version": "experiment015-direct-rounding-sga-v2",
        "method": method,
        "source_graph_sha256": graph_sha,
        "source_factor_sha256": factor_sha,
        "restart_indices": [int(i) for i in start_indices],
        "settings": settings,
        "optimization_variables": "rounding normal r only; EVA Y frozen",
    }
    key = hashlib.sha256(json.dumps(
        config, sort_keys=True, allow_nan=False,
    ).encode("utf8")).hexdigest()[:20]
    return key, config


def _verify(record, config):
    if record.get("status") != "complete" or record.get("config") != config:
        raise ValueError("Cached direct-rounding configuration mismatch")
    archive = Path(record["archive_path"])
    report = verify_archive(archive)
    n = len(config["restart_indices"])
    if (report["status"] != "complete"
            or report["graph_sha256"] != config["source_graph_sha256"]
            or len(record["trials"]) != n):
        raise ValueError("Saved direct-rounding partition archive invalid")
    for stage in STAGES:
        method = f"Exp015-{stage}_EVA"
        if report["methods"][method]["verified_partitions"] != n:
            raise ValueError("Missing saved direct-rounding stage partitions")
        parts, cuts = load_method(archive/f"{method}.npz")
        if not np.allclose(
            cuts, [trial["scores"][stage] for trial in record["trials"]],
            atol=1e-8, rtol=0,
        ):
            raise ValueError("Cached Max-Cut scores mismatch stored partitions")
        # Check direction/partition consistency below for raw stages.
    directions = Path(record["directions_path"])
    if _file_sha256(directions) != record["directions_sha256"]:
        raise ValueError("Saved rounding normals SHA mismatch")
    with np.load(directions, allow_pickle=False) as saved:
        old = saved["original_normals"]
        optimized = saved["optimized_normals"]
        restarts = saved["restart_indices"]
    if (old.shape != optimized.shape
            or old.shape != (n, _Y.shape[1])
            or not np.array_equal(
                restarts.astype(int), config["restart_indices"],
            )
            or not np.allclose(np.linalg.norm(old, axis=1), 1, atol=1e-8)
            or not np.allclose(np.linalg.norm(optimized, axis=1), 1, atol=1e-8)):
        raise ValueError("Cached optimized rounding normals invalid")
    old_parts, _ = load_method(archive/"Exp015-GWRaw_EVA.npz")
    optimized_parts, _ = load_method(archive/"Exp015-OptimizedRaw_EVA.npz")
    for i in range(n):
        if (not np.array_equal(old_parts[i], (_Y@old[i] > 0).astype(np.int8))
                or not np.array_equal(optimized_parts[i], (
                    _Y@optimized[i] > 0
                ).astype(np.int8))):
            raise ValueError("Cached raw partition not produced by archived r")
    for trial, index in zip(record["trials"], config["restart_indices"]):
        scores = trial["scores"]
        if trial["restart"] != index or not (
            scores["OptimizedRaw"]+1e-8 >= scores["GWRaw"]
            and scores["GWRawKL"]+1e-8 >= scores["GWRaw"]
            and scores["OptimizedRawKL"]+1e-8 >= scores["OptimizedRaw"]
        ):
            raise ValueError("Saved direct-rounding refinement decreased cut")
    return record


def _solve(job):
    config = job["config"]
    case_file = Path(job["case_path"])
    if case_file.is_file():
        return _verify(
            json.loads(case_file.read_text(encoding="utf8")), config
        ), True
    if _GRAPH is None or _Y is None or _SOURCE != [
        config["source_graph_sha256"], config["source_factor_sha256"],
    ]:
        raise RuntimeError("GW graph and frozen EVA factor not initialized")
    method = config["method"]
    settings = config["settings"]
    graph_num = int(Path(job["graph_path"]).stem[1:])
    parts = {stage: [] for stage in STAGES}
    cuts = {stage: [] for stage in STAGES}
    raw_normals, new_normals, trials = [], [], []

    for restart in config["restart_indices"]:
        # Same original FJ_EVA restart convention. Normalization does
        # not alter a nonzero homogeneous hyperplane's binary labels.
        baseline_rng = np.random.default_rng(np.random.SeedSequence([
            settings["rounding_seed"], graph_num, 2, restart, 0,
        ]))
        normal = baseline_rng.standard_normal(_Y.shape[1])
        normal /= np.linalg.norm(normal)
        raw = (_Y@normal > 0).astype(np.int8)
        raw_score = float(cut_weight(_EDGES, raw))
        # Hybrid runs start from precisely the SAME SGA random stream
        # as their nonhybrid counterparts. Only the exact sweep is new.
        sga_code = 151 if method.startswith("es-sga") else 152
        stochastic_rng = np.random.default_rng(np.random.SeedSequence([
            settings["search_seed"], graph_num, restart, 15015, sga_code,
        ]))
        if method == "linesweep":
            new_normal, new_raw, search = optimize_hyperplane(
                _GRAPH, _Y, stochastic_rng, initial=normal,
                directions=settings["line_directions"],
                target=settings["target"],
            )
            search_history = {"linesweep": search}
        elif method.startswith("es-sga"):
            new_normal, new_raw, sga = optimize_es_sga(
                _GRAPH, _Y, stochastic_rng, initial=normal,
                steps=settings["sga_steps"],
                population=settings["sga_population"],
                sigma_start=settings["sigma_start"],
                sigma_end=settings["sigma_end"],
                learning_rate=settings["learning_rate"],
                restart_patience=settings["restart_patience"],
                target=settings["target"],
            )
            search_history = {"es_sga": sga}
        elif method.startswith("soft-sga"):
            new_normal, new_raw, sga = optimize_soft_sga(
                _GRAPH, _Y, stochastic_rng, initial=normal,
                steps=settings["sga_steps"],
                edge_batch=settings["edge_batch"],
                temperature_start=settings["temperature_start"],
                temperature_end=settings["temperature_end"],
                learning_rate=settings["learning_rate"],
                restart_patience=settings["restart_patience"],
                target=settings["target"],
            )
            search_history = {"soft_sga": sga}
        else:
            raise RuntimeError("Invalid direct-rounding optimizer")
        if method.endswith("-sweep"):
            # Exact line search is run AFTER the SGD/Adam search.
            sweep_rng = np.random.default_rng(np.random.SeedSequence([
                settings["search_seed"], graph_num, restart, 15015, 153,
            ]))
            new_normal, new_raw, sweep = optimize_hyperplane(
                _GRAPH, _Y, sweep_rng, initial=new_normal,
                directions=settings["line_directions"],
                target=settings["target"],
            )
            search_history["linesweep"] = sweep
        new_raw_score = float(cut_weight(_EDGES, new_raw))
        if new_raw_score + 1e-8 < raw_score:
            raise AssertionError("Direct rounding search lost its source GW cut")

        def refine(labels):
            best_labels = labels.copy()
            best_cut = float(cut_weight(_EDGES, labels))
            attempts = []
            for rep in range(settings["kl_repeats"]):
                rng = np.random.default_rng(np.random.SeedSequence([
                    settings["search_seed"], graph_num, restart, 101, rep,
                ]))
                result, details = optimize_kl_fast(
                    _GRAPH, labels.copy(), rng,
                    passes=settings["kl_passes"],
                )
                cut = float(cut_weight(_EDGES, result))
                attempts.append(cut)
                if cut > best_cut:
                    best_labels, best_cut = result.copy(), cut
            return best_labels, best_cut, attempts

        old_kl, old_kl_score, old_runs = refine(raw)
        new_kl, new_kl_score, new_runs = refine(new_raw)
        stage_parts = {
            "GWRaw": raw, "OptimizedRaw": new_raw,
            "GWRawKL": old_kl, "OptimizedRawKL": new_kl,
        }
        stage_scores = {
            "GWRaw": raw_score, "OptimizedRaw": new_raw_score,
            "GWRawKL": old_kl_score, "OptimizedRawKL": new_kl_score,
        }
        for stage in STAGES:
            parts[stage].append(stage_parts[stage])
            cuts[stage].append(float(stage_scores[stage]))
        raw_normals.append(normal)
        new_normals.append(new_normal)
        trials.append({
            "restart": int(restart),
            "method": method, "scores": stage_scores,
            "raw_gain": new_raw_score-raw_score,
            "after_KL_gain_vs_raw_control": new_kl_score-old_kl_score,
            "search": search_history,
            "old_KL_runs": old_runs, "optimized_KL_runs": new_runs,
        })

    # Independent saved normals, the exact optimized decision variables.
    direction_path = Path(job["directions_path"])
    _atomic_npz(
        direction_path,
        original_normals=np.asarray(raw_normals, dtype=np.float64),
        optimized_normals=np.asarray(new_normals, dtype=np.float64),
        restart_indices=np.asarray(
            config["restart_indices"], dtype=np.int32,
        ),
    )
    norm_sha = _file_sha256(direction_path)
    trial_path = Path(job["trial_path"])
    archive = new_archive(
        Path(job["output_dir"])/"archives", new_run_id(),
        Path(job["graph_path"]), _GRAPH.shape[0], _EDGES,
        2, config, jsonl_output=trial_path,
        argv=["experiment015", method],
    )
    for stage in STAGES:
        save_method(
            archive, f"Exp015-{stage}_EVA",
            parts[stage], cuts[stage], k=2,
        )
    update_manifest(archive, status="complete", solver_info={
        "optimized_variables": "original EVA hyperplane normal ONLY",
        "factor_immutable": True, "no_kmeans": True,
        "method": method,
        "directions_path": str(direction_path),
        "directions_sha256": norm_sha,
        "original_graph_signed_weighted_MaxCut": True,
    })
    verified = verify_archive(archive["path"])
    if (verified["status"] != "complete"
            or verified["graph_sha256"] != config["source_graph_sha256"]
            or any(verified["methods"][f"Exp015-{stage}_EVA"][
                       "verified_partitions"] != len(trials)
                   for stage in STAGES)):
        raise RuntimeError("Fresh direct-rounding partition archive failed")
    record = {
        "status": "complete",
        "graph": Path(job["graph_path"]).stem,
        "id": job["id"],
        "method": method,
        "config": config,
        "archive_path": str(archive["path"]),
        "directions_path": str(direction_path),
        "directions_sha256": norm_sha,
        "trial_path": str(trial_path),
        "trials": trials,
        "best_by_stage": {
            stage: float(max(cuts[stage])) for stage in STAGES
        },
    }
    trial_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = trial_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf8") as stream:
        for trial in trials:
            stream.write(json.dumps({
                "graph": record["graph"],
                "id": record["id"],
                "method": method,
                "archive_path": record["archive_path"],
                **trial,
            }, sort_keys=True, allow_nan=False)+"\n")
    os.replace(temporary, trial_path)
    _write_json(case_file, record)
    return record, False


def _summaries(dest, *, target):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest/"summary.csv"
    temp = path.with_suffix(".tmp")
    fields = [
        "graph", "method", "restart", "GWRaw",
        "OptimizedRaw", "GWRawKL", "OptimizedRawKL",
        "raw_gain", "KL_gain_vs_control", "gap_to_target",
        "archive_path", "directions_path",
    ]
    with temp.open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case_file in sorted((dest/"cases").glob("G*.json")):
            record = json.loads(case_file.read_text(encoding="utf8"))
            for trial in record["trials"]:
                writer.writerow({
                    "graph": record["graph"], "method": record["method"],
                    "restart": trial["restart"],
                    **trial["scores"],
                    "raw_gain": trial["raw_gain"],
                    "KL_gain_vs_control": trial["after_KL_gain_vs_raw_control"],
                    "gap_to_target": target-trial["scores"]["OptimizedRawKL"],
                    "archive_path": record["archive_path"],
                    "directions_path": record["directions_path"],
                })
    os.replace(temp, path)
    return path


def run_graph(
    graph_path, *, factor_dir, dest, methods=METHODS,
    first=0, restarts=12, chunk_size=3, workers=4,
    rounding_seed=2026, search_seed=2026,
    line_directions=60, sga_steps=300, sga_population=16,
    sigma_start=.3, sigma_end=.025,
    temperature_start=.5, temperature_end=.04,
    edge_batch=2048, learning_rate=.08,
    restart_patience=80, kl_passes=30, kl_repeats=2,
    target=27047.,
):
    graph_path, dest = Path(graph_path), Path(dest)
    if (not methods or any(m not in METHODS for m in methods)
            or len(set(methods)) != len(methods)
            or first < 0 or restarts < 1 or chunk_size < 1
            or workers < 1 or line_directions < 1 or sga_steps < 1
            or sga_population < 1 or edge_batch < 1
            or kl_passes < 1 or kl_repeats < 1
            or restart_patience < 1 or rounding_seed < 0
            or search_seed < 0 or not np.isfinite(target)
            or min(sigma_start,sigma_end,temperature_start,
                   temperature_end,learning_rate) <= 0):
        raise ValueError("Invalid direct hyperplane rounding/SGA budgets")
    graph, y, _, graph_sha, factor_sha = _factor_and_case(
        graph_path, Path(factor_dir),
    )
    if dest.resolve() == Path(factor_dir).resolve():
        raise ValueError("Output must differ from original EVA factor directory")
    dest.mkdir(parents=True, exist_ok=True)
    _immutable_factor_copy(Path(factor_dir), dest, graph_path.stem, factor_sha)
    settings = {
        "rounding_seed": rounding_seed, "search_seed": search_seed,
        "line_directions": line_directions,
        "sga_steps": sga_steps, "sga_population": sga_population,
        "sigma_start": sigma_start, "sigma_end": sigma_end,
        "temperature_start": temperature_start,
        "temperature_end": temperature_end,
        "edge_batch": edge_batch,
        "learning_rate": learning_rate,
        "restart_patience": restart_patience,
        "kl_passes": kl_passes, "kl_repeats": kl_repeats,
        "target": target,
    }
    jobs = []
    for offset in range(first, first+restarts, chunk_size):
        indices = list(range(offset, min(first+restarts, offset+chunk_size)))
        for method in methods:
            key, config = _identity(
                method, indices, graph_sha, factor_sha, settings=settings,
            )
            basename = f"{graph_path.stem}-{method}-{key}"
            jobs.append({
                "id": key, "config": config, "graph_path": str(graph_path),
                "output_dir": str(dest),
                "case_path": str(dest/"cases"/f"{basename}.json"),
                "directions_path": str(dest/"normals"/f"{basename}.npz"),
                "trial_path": str(dest/"trials"/f"{basename}.jsonl"),
            })
    print(
        f"{graph_path.stem}: verified frozen EVA Y={y.shape}, "
        f"{len(methods)} direct-rounding methods x {restarts} paired "
        f"raw GW hyperplane starts, {len(jobs)} resumable jobs; "
        f"workers={workers}; target={target:g}",
        flush=True,
    )
    results, errors = [], []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init,
        initargs=(str(graph_path), str(factor_dir), [graph_sha, factor_sha]),
    ) as pool:
        futures = {pool.submit(_solve, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                record, cached = future.result()
                results.append(record)
                best = record["best_by_stage"]
                print(
                    f"{record['graph']} {record['method']} "
                    f"{'verified skip' if cached else 'complete'} "
                    f"n={len(record['trials'])}: raw "
                    f"{best['GWRaw']:g} -> {best['OptimizedRaw']:g}; "
                    f"KL {best['GWRawKL']:g} -> {best['OptimizedRawKL']:g}; "
                    f"gap to {target:g}: "
                    f"{target-max(best.values()):+g}",
                    flush=True,
                )
            except Exception as exc:
                diagnostic = {
                    "graph": graph_path.stem,
                    "method": job["config"]["method"],
                    "case_path": job["case_path"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc().replace(
                        str(Path.home()), "<HOME>",
                    ),
                }
                errors.append(diagnostic)
                _write_json(
                    dest/"errors"/f"{Path(job['case_path']).stem}.json",
                    diagnostic,
                )
                print(
                    f"{graph_path.stem} {job['config']['method']} FAILED: "
                    f"{exc}", flush=True,
                )
            _summaries(dest, target=target)
    if errors:
        raise RuntimeError(
            f"{len(errors)} methods/chunks failed. See {dest/'errors'}"
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=["G63"])
    parser.add_argument("--gset-dir", type=Path, default=Path("data"))
    parser.add_argument("--factor-dir", type=Path, default=Path(
        "results/gset/experiment003-eva-numba",
    ))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "results/gset/experiment015-direct-rounding",
    ))
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS))
    parser.add_argument("--start-restart", type=int, default=0)
    parser.add_argument("--restarts", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rounding-seed", type=int, default=2026)
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument("--line-directions", type=int, default=60)
    parser.add_argument("--sga-steps", type=int, default=300)
    parser.add_argument("--sga-population", type=int, default=16)
    parser.add_argument("--sigma-start", type=float, default=.3)
    parser.add_argument("--sigma-end", type=float, default=.025)
    parser.add_argument("--temperature-start", type=float, default=.5)
    parser.add_argument("--temperature-end", type=float, default=.04)
    parser.add_argument("--edge-batch", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=.08)
    parser.add_argument("--restart-patience", type=int, default=80)
    parser.add_argument("--kl-passes", type=int, default=30)
    parser.add_argument("--kl-repeats", type=int, default=2)
    parser.add_argument("--target", type=float, default=27047.)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--push-branch")
    args = parser.parse_args()
    if args.push and not args.push_branch:
        parser.error("--push requires --push-branch")
    if args.output_dir.resolve() == args.factor_dir.resolve():
        parser.error("Output must differ from completed original EVA source")
    graphs = discover_graphs(args.gset_dir, args.graphs)
    failures = 0
    for path in graphs:
        try:
            run_graph(
                path, factor_dir=args.factor_dir,
                dest=args.output_dir, methods=tuple(args.methods),
                first=args.start_restart, restarts=args.restarts,
                chunk_size=args.chunk_size, workers=args.workers,
                rounding_seed=args.rounding_seed,
                search_seed=args.search_seed,
                line_directions=args.line_directions,
                sga_steps=args.sga_steps,
                sga_population=args.sga_population,
                sigma_start=args.sigma_start, sigma_end=args.sigma_end,
                temperature_start=args.temperature_start,
                temperature_end=args.temperature_end,
                edge_batch=args.edge_batch,
                learning_rate=args.learning_rate,
                restart_patience=args.restart_patience,
                kl_passes=args.kl_passes,
                kl_repeats=args.kl_repeats,
                target=args.target,
            )
            (args.output_dir/"errors"/f"{path.stem}.json").unlink(
                missing_ok=True,
            )
        except (OSError, ValueError, KeyError, RuntimeError,
                AssertionError, ArithmeticError) as exc:
            failures += 1
            _write_json(args.output_dir/"errors"/f"{path.stem}.json", {
                "graph": path.stem,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().replace(
                    str(Path.home()), "<HOME>",
                ),
            })
            print(f"{path.stem}: ERROR {exc}", flush=True)
            if not args.continue_on_error:
                if args.push:
                    _git_push(
                        args.output_dir, "experiment015 partial direct rounding",
                        allowed_branch=args.push_branch,
                    )
                raise
    if args.push:
        _git_push(
            args.output_dir, "experiment015 optimized GW normals",
            allowed_branch=args.push_branch,
        )
    print(
        f"Experiment015: {len(graphs)} graphs, errors={failures}; "
        f"summary={args.output_dir/'summary.csv'}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
