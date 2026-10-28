"""Experiment014: reoptimize EVA factor for expected raw GW rounding, NO K-means.

Independent experiment axes: epsilon (ordinary or Gaussian-noisy GW),
factor tangent jitter, gradient step, and optimization random seed.
Every config reads the SAME completed original EVA factor/graph,
maximizes its expected signed GW cut, and archives the best new factor.

Each config samples the SAME paired raw hyperplane directions on
original and optimized factors, and then KL-refines the SAME highest
scoring paired draws on both sides. Bit-packed original graph partitions
and the new optimized n*p factor are SHA-verified and committed to the
repo; completed configurations are resumable with exact settings.
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
    _atomic_npz, _file_sha256, load_method, new_archive,
    new_run_id, save_method, update_manifest, verify_archive,
)
from .eva_cli import _git_push, _write_json
from .eva_rounding_cli import _factor_and_case, _immutable_factor_copy
from .eva_gw_expected import (
    gw_expected_cut, optimize_gw_factor, raw_gw_pair,
)
from .eva_elite_search import optimize_kl_fast
from .io import edge_arrays
from .objective import cut_weight
from .sweep import discover_graphs


_GRAPH = None
_Y = None
_EDGES = None
_HASHES = None
STAGES = ("OldRawGW", "NewRawGW", "OldRawGWKL", "NewRawGWKL")


def _init(path, factor_dir, hashes):
    global _GRAPH, _Y, _EDGES, _HASHES
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    graph, y, _, graph_sha, factor_sha = _factor_and_case(
        Path(path), Path(factor_dir),
    )
    if [graph_sha, factor_sha] != hashes:
        raise ValueError("Original EVA source factor/graph SHA mismatch")
    _GRAPH, _Y, _EDGES = graph, y, edge_arrays(graph)
    _HASHES = [graph_sha, factor_sha]


def _identity(graph_sha, factor_sha, *, epsilon, jitter, step,
              optimize_seed, rounding_seed, max_sweeps, patience,
              num_rounds, kl_top, kl_passes):
    settings = {
        "version": "experiment014-expected-gw-v1",
        "graph_sha256": graph_sha, "source_factor_sha256": factor_sha,
        "epsilon": float(epsilon), "jitter": float(jitter),
        "step": float(step), "optimize_seed": int(optimize_seed),
        "rounding_seed": int(rounding_seed),
        "max_sweeps": int(max_sweeps), "patience": int(patience),
        "num_rounds": int(num_rounds), "kl_top": int(kl_top),
        "kl_passes": int(kl_passes),
        "raw_rounding": "paired Gaussian GW, no K-means",
        "objective": "exact expected original signed weighted cut",
    }
    key = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf8"),
    ).hexdigest()[:20]
    return key, settings


def _verify_case(record, settings):
    if record.get("status") != "complete" or record.get("settings") != settings:
        raise ValueError("Saved GW expected refinement does not match settings")
    factor = Path(record["optimized_factor"])
    if _file_sha256(factor) != record["optimized_factor_sha256"]:
        raise ValueError("Saved second-stage factor checksum mismatch")
    with np.load(factor, allow_pickle=False) as stored:
        y = stored["vectors"]
        metadata = json.loads(str(stored["metadata"].item()))
    if (metadata != settings or
            not np.allclose(np.linalg.norm(y, axis=1), 1, atol=1e-7)):
        raise ValueError("Optimized GW factor metadata/row norm mismatch")
    checked = verify_archive(record["archive_path"])
    if (checked["status"] != "complete"
            or checked["graph_sha256"] != settings["graph_sha256"]):
        raise ValueError("Saved GW archive graph verification failed")
    counts = {
        "OldRawGW": settings["num_rounds"],
        "NewRawGW": settings["num_rounds"],
        "OldRawGWKL": settings["kl_top"],
        "NewRawGWKL": settings["kl_top"],
    }
    for stage, count in counts.items():
        method = f"Exp014-{stage}_EVA"
        if checked["methods"][method]["verified_partitions"] != count:
            raise ValueError(f"Saved GW {stage} partition count mismatch")
        _, scores = load_method(
            Path(record["archive_path"]) / f"{method}.npz",
        )
        if not np.allclose(scores, record["scores"][stage],
                           atol=1e-8, rtol=0):
            raise ValueError("GW stored scores mismatch partition archive")
    return record


def _run(job):
    case_path = Path(job["case_path"])
    settings = job["settings"]
    if case_path.is_file():
        return _verify_case(
            json.loads(case_path.read_text(encoding="utf8")), settings,
        ), True
    if _GRAPH is None or _Y is None or _HASHES != [
        settings["graph_sha256"], settings["source_factor_sha256"],
    ]:
        raise RuntimeError("Factor/graph worker initialization mismatch")

    graph_num = int(Path(job["graph_path"]).stem[1:])
    rng = np.random.default_rng(np.random.SeedSequence([
        settings["optimize_seed"], graph_num, 14014,
    ]))
    y_new, optimization = optimize_gw_factor(
        _GRAPH, _Y, rng, epsilon=settings["epsilon"],
        max_sweeps=settings["max_sweeps"],
        step=settings["step"], jitter=settings["jitter"],
        patience=settings["patience"],
    )
    before_expected = gw_expected_cut(
        _EDGES, _Y, epsilon=settings["epsilon"],
    )
    after_expected = gw_expected_cut(
        _EDGES, y_new, epsilon=settings["epsilon"],
    )
    if (after_expected + 1e-7 < before_expected
            or not np.isclose(after_expected, optimization["optimized_expected_cut"],
                              atol=1e-6, rtol=0)):
        raise RuntimeError("GW expected optimization decreased expected cut")

    labels_by_stage = {stage: [] for stage in STAGES}
    cuts_by_stage = {stage: [] for stage in STAGES}
    raw_draws = []
    for draw in range(settings["num_rounds"]):
        draw_rng = np.random.default_rng(np.random.SeedSequence([
            settings["rounding_seed"], graph_num, 2, draw, 14014,
        ]))
        old, new = raw_gw_pair(
            _Y, y_new, draw_rng, epsilon=settings["epsilon"],
        )
        old_cut = float(cut_weight(_EDGES, old))
        new_cut = float(cut_weight(_EDGES, new))
        labels_by_stage["OldRawGW"].append(old)
        labels_by_stage["NewRawGW"].append(new)
        cuts_by_stage["OldRawGW"].append(old_cut)
        cuts_by_stage["NewRawGW"].append(new_cut)
        raw_draws.append({
            "draw": draw, "old_cut": old_cut, "new_cut": new_cut,
            "pair_difference": new_cut-old_cut,
        })
    # Compare KL on exactly the SAME selected draw indices. Selection
    # by max(old raw, new raw) avoids favoring either factor exclusively.
    pair_peak = np.maximum(
        cuts_by_stage["OldRawGW"], cuts_by_stage["NewRawGW"],
    )
    selected = np.argsort(-np.asarray(pair_peak),
                          kind="stable")[:settings["kl_top"]]
    kl_trials = []
    for draw in selected:
        control_rng = np.random.default_rng(np.random.SeedSequence([
            settings["rounding_seed"], graph_num, int(draw), 101, 14014,
        ]))
        optimized_rng = np.random.default_rng(np.random.SeedSequence([
            settings["rounding_seed"], graph_num, int(draw), 101, 14014,
        ]))
        old, old_diag = optimize_kl_fast(
            _GRAPH, labels_by_stage["OldRawGW"][draw], control_rng,
            passes=settings["kl_passes"],
        )
        new, new_diag = optimize_kl_fast(
            _GRAPH, labels_by_stage["NewRawGW"][draw], optimized_rng,
            passes=settings["kl_passes"],
        )
        for stage, labels in (("OldRawGWKL", old), ("NewRawGWKL", new)):
            labels_by_stage[stage].append(labels)
            cuts_by_stage[stage].append(float(cut_weight(_EDGES, labels)))
        if (cuts_by_stage["OldRawGWKL"][-1]+1e-8 <
                cuts_by_stage["OldRawGW"][draw]
                or cuts_by_stage["NewRawGWKL"][-1]+1e-8 <
                cuts_by_stage["NewRawGW"][draw]):
            raise AssertionError("KL decreased raw original-graph cut")
        kl_trials.append({
            "draw": int(draw),
            "old_raw": cuts_by_stage["OldRawGW"][draw],
            "new_raw": cuts_by_stage["NewRawGW"][draw],
            "old_KL": cuts_by_stage["OldRawGWKL"][-1],
            "new_KL": cuts_by_stage["NewRawGWKL"][-1],
            "old_KL_seconds": old_diag["seconds"],
            "new_KL_seconds": new_diag["seconds"],
        })

    # Store the NEW unit-row factor independently from the immutable
    # completed original EVA checkpoint. It is NEVER written into
    # experiment003 source directories.
    factor_path = Path(job["factor_path"])
    _atomic_npz(
        factor_path, vectors=y_new,
        metadata=np.asarray(json.dumps(settings, sort_keys=True)),
    )
    factor_sha = _file_sha256(factor_path)
    archive = new_archive(
        Path(job["output_dir"])/"archives", new_run_id(),
        Path(job["graph_path"]), _GRAPH.shape[0], _EDGES,
        2, settings, jsonl_output=Path(job["trial_path"]),
        argv=["experiment014", job["case_id"]],
    )
    for stage in STAGES:
        save_method(
            archive, f"Exp014-{stage}_EVA",
            labels_by_stage[stage], cuts_by_stage[stage], k=2,
        )
    update_manifest(archive, status="complete", solver_info={
        "reoptimized_analytical_expected_cut": True,
        "original_graph_signed": True,
        "no_kmeans": True,
        "optimized_factor_path": str(factor_path),
        "optimized_factor_sha256": factor_sha,
        "source_expected_cut": float(before_expected),
        "optimized_expected_cut": float(after_expected),
        "best_cut_by_stage": {
            stage: float(max(cuts_by_stage[stage])) for stage in STAGES
        },
    })
    verified = verify_archive(archive["path"])
    if (verified["status"] != "complete"
            or verified["graph_sha256"] != settings["graph_sha256"]
            or any(verified["methods"][f"Exp014-{stage}_EVA"][
                       "verified_partitions"] != len(labels_by_stage[stage])
                   for stage in STAGES)):
        raise RuntimeError("New GW partitions failed independent verification")
    record = {
        "status": "complete", "graph": Path(job["graph_path"]).stem,
        "id": job["case_id"], "settings": settings,
        "optimization": optimization,
        "source_expected_cut": float(before_expected),
        "optimized_expected_cut": float(after_expected),
        "optimized_factor": str(factor_path),
        "optimized_factor_sha256": factor_sha,
        "archive_path": str(archive["path"]),
        "trial_path": job["trial_path"],
        "raw_draws": raw_draws,
        "kl_trials": kl_trials,
        "scores": cuts_by_stage,
        "best_cut_by_stage": {
            stage: float(max(cuts_by_stage[stage])) for stage in STAGES
        },
    }
    trial_path = Path(job["trial_path"])
    trial_path.parent.mkdir(parents=True, exist_ok=True)
    temp = trial_path.with_suffix(".tmp")
    with temp.open("w", encoding="utf8") as stream:
        for row in raw_draws:
            stream.write(json.dumps({
                "stage": "raw_GW", "config": job["case_id"],
                "graph_sha256": settings["graph_sha256"], **row,
            }, allow_nan=False, sort_keys=True)+"\n")
        for row in kl_trials:
            stream.write(json.dumps({
                "stage": "paired_KL", "config": job["case_id"],
                "graph_sha256": settings["graph_sha256"], **row,
            }, allow_nan=False, sort_keys=True)+"\n")
    os.replace(temp, trial_path)
    _write_json(case_path, record)
    return record, False


def _summary(dest, *, target):
    dest = Path(dest)
    output = dest / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "graph", "id", "epsilon", "jitter", "step", "optimize_seed",
        "max_sweeps", "rounds", "kl_top", "kl_passes",
        "source_expected_cut", "optimized_expected_cut",
        "expected_gain", "old_raw_best", "new_raw_best",
        "old_KL_best", "new_KL_best", "best_feasible",
        "gap_to_target", "factor_path", "archive_path",
    )
    temp = output.with_suffix(".tmp")
    with temp.open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for case_file in sorted((dest / "cases").glob("G*.json")):
            rec = json.loads(case_file.read_text(encoding="utf8"))
            settings = rec["settings"]
            maxima = rec["best_cut_by_stage"]
            top = max(maxima.values())
            writer.writerow({
                "graph": rec["graph"], "id": rec["id"],
                "epsilon": settings["epsilon"],
                "jitter": settings["jitter"], "step": settings["step"],
                "optimize_seed": settings["optimize_seed"],
                "max_sweeps": settings["max_sweeps"],
                "rounds": settings["num_rounds"],
                "kl_top": settings["kl_top"],
                "kl_passes": settings["kl_passes"],
                "source_expected_cut": rec["source_expected_cut"],
                "optimized_expected_cut": rec["optimized_expected_cut"],
                "expected_gain": rec["optimized_expected_cut"]-rec["source_expected_cut"],
                "old_raw_best": maxima["OldRawGW"],
                "new_raw_best": maxima["NewRawGW"],
                "old_KL_best": maxima["OldRawGWKL"],
                "new_KL_best": maxima["NewRawGWKL"],
                "best_feasible": top,
                "gap_to_target": target-top,
                "factor_path": rec["optimized_factor"],
                "archive_path": rec["archive_path"],
            })
    os.replace(temp, output)
    return output


def run_graph(
    path, *, factor_dir, dest, epsilons=(0.,),
    jitters=(0., 0.05), steps=(0.05, 0.15),
    optimize_seeds=(2026, 2027),
    num_rounds=500, kl_top=20, kl_passes=25,
    max_sweeps=50, patience=12,
    rounding_seed=2026, workers=3, target=27047.,
):
    path, factor_dir, dest = Path(path), Path(factor_dir), Path(dest)
    if (not epsilons or not jitters or not steps or not optimize_seeds
            or num_rounds < 1 or kl_top < 1 or kl_top > num_rounds
            or kl_passes < 1 or max_sweeps < 1 or patience < 1
            or workers < 1 or rounding_seed < 0
            or not np.isfinite(target)
            or any(not 0 <= v < 1 for v in epsilons)
            or any(v < 0 for v in jitters)
            or any(v <= 0 for v in steps)
            or any(v < 0 for v in optimize_seeds)):
        raise ValueError("Invalid matrix optimization/grid/rounding parameters")
    graph, factor, _, graph_sha, factor_sha = _factor_and_case(
        path, factor_dir,
    )
    if dest.resolve() == factor_dir.resolve():
        raise ValueError("Refuse to overwrite original EVA factor directory")
    dest.mkdir(parents=True, exist_ok=True)
    _immutable_factor_copy(factor_dir, dest, path.stem, factor_sha)
    jobs = []
    signatures = set()
    for epsilon in epsilons:
        for jitter in jitters:
            for step in steps:
                for optimize_seed in optimize_seeds:
                    key, settings = _identity(
                        graph_sha, factor_sha, epsilon=epsilon,
                        jitter=jitter, step=step,
                        optimize_seed=optimize_seed,
                        rounding_seed=rounding_seed,
                        max_sweeps=max_sweeps, patience=patience,
                        num_rounds=num_rounds, kl_top=kl_top,
                        kl_passes=kl_passes,
                    )
                    if key in signatures:
                        continue
                    signatures.add(key)
                    jobs.append({
                        "case_id": key, "settings": settings,
                        "graph_path": str(path),
                        "output_dir": str(dest),
                        "case_path": str(dest/"cases"/f"{path.stem}-{key}.json"),
                        "factor_path": str(dest/"optimized-factors"/f"{path.stem}-{key}.npz"),
                        "trial_path": str(dest/"trials"/f"{path.stem}-{key}.jsonl"),
                    })
    print(
        f"{path.stem}: verified source EVA factor shape={factor.shape}; "
        f"{len(jobs)} GW-expectation optimizer configs; "
        f"{num_rounds} paired raw hyperplanes + {kl_top} matched KL draws "
        f"each; workers={workers}; target={target:g}",
        flush=True,
    )
    results, errors = [], []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init,
        initargs=(str(path), str(factor_dir), [graph_sha, factor_sha]),
    ) as executor:
        futures = {executor.submit(_run, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                record, skipped = future.result()
                results.append(record)
                cuts = record["best_cut_by_stage"]
                maximum = max(cuts.values())
                print(
                    f"{path.stem} expectedGW eps={record['settings']['epsilon']:g} "
                    f"jitter={record['settings']['jitter']:g} "
                    f"step={record['settings']['step']:g} "
                    f"seed={record['settings']['optimize_seed']} "
                    f"{'verified skip' if skipped else 'complete'}: "
                    f"E[cut] {record['source_expected_cut']:.2f} -> "
                    f"{record['optimized_expected_cut']:.2f}; "
                    f"raw {cuts['OldRawGW']:g} -> {cuts['NewRawGW']:g}; "
                    f"KL {cuts['OldRawGWKL']:g} -> {cuts['NewRawGWKL']:g}; "
                    f"best={maximum:g}, gap={target-maximum:+g}",
                    flush=True,
                )
            except Exception as exc:
                errors.append({
                    "graph": path.stem,
                    "case": job["case_path"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc().replace(
                        str(Path.home()), "<HOME>",
                    ),
                })
                _write_json(
                    dest/"errors"/f"{path.stem}-{job['case_id']}.json",
                    errors[-1],
                )
                print(f"{path.stem} GW refiner FAILED: {exc}", flush=True)
            _summary(dest, target=target)
    if errors:
        raise RuntimeError(
            f"{len(errors)} GW refinement configs failed; see {dest/'errors'}"
        )
    print(
        f"{path.stem} GW expectation grid completed: "
        f"best feasible={max(max(x['best_cut_by_stage'].values()) for x in results):g}; "
        f"summary={dest/'summary.csv'}",
        flush=True,
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
        "results/gset/experiment014-gw-expected",
    ))
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0., .02])
    parser.add_argument("--jitters", type=float, nargs="+", default=[0., .05])
    parser.add_argument("--steps", type=float, nargs="+", default=[.05, .15])
    parser.add_argument("--optimize-seeds", type=int, nargs="+",
                        default=[2026, 2027])
    parser.add_argument("--max-sweeps", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--kl-top", type=int, default=20)
    parser.add_argument("--kl-passes", type=int, default=25)
    parser.add_argument("--rounding-seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--target", type=float, default=27047.)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--push-branch")
    args = parser.parse_args()
    if args.push and not args.push_branch:
        parser.error("--push requires --push-branch")
    if args.output_dir.resolve() == args.factor_dir.resolve():
        parser.error("Output directory must differ from the original factor dir")
    paths = discover_graphs(args.gset_dir, args.graphs)
    errors = 0
    for path in paths:
        try:
            run_graph(
                path, factor_dir=args.factor_dir, dest=args.output_dir,
                epsilons=tuple(args.epsilons), jitters=tuple(args.jitters),
                steps=tuple(args.steps),
                optimize_seeds=tuple(args.optimize_seeds),
                num_rounds=args.rounds, kl_top=args.kl_top,
                kl_passes=args.kl_passes,
                max_sweeps=args.max_sweeps, patience=args.patience,
                rounding_seed=args.rounding_seed, workers=args.workers,
                target=args.target,
            )
        except (OSError, ValueError, KeyError, RuntimeError,
                ArithmeticError, AssertionError) as exc:
            errors += 1
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
                        args.output_dir, "experiment014 partial expected GW factor",
                        allowed_branch=args.push_branch,
                    )
                raise
    if args.push:
        _git_push(
            args.output_dir, "experiment014 expected GW factor results",
            allowed_branch=args.push_branch,
        )
    print(
        f"Experiment014: {len(paths)} graph(s), errors={errors}; "
        f"summary={args.output_dir/'summary.csv'}",
        flush=True,
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
