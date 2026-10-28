"""Experiment011: extensive parallel elite diversification of VERIFIED KL cuts.

Source is experiment010's archived KL partitions, not regenerated
roundings. It operates on the original signed weighted G-set graph.

Three concurrently scheduled methods (ILS-KL, EliteMix-KL, Tabu-KL)
share the same top-score distinct KL starting partitions. Independent
per-(source,replicate,method) deterministic seeds are unaffected by
scheduling and worker count. Per-chunk bit-packed archives independently
re-score ALL outputs and preserve complete per-cycle traces.
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
from .eva_elite_search import METHODS, optimize_elite
from .io import edge_arrays, load_graph
from .objective import cut_weight
from .sweep import discover_graphs


_W_GRAPH = None
_W_EDGES = None
_W_ELITES = None


def _source_kl(path, source_dir, *, source_method="K-MeansNM-Flip_EVA",
               source_seed=2026, source_passes=30, factor_prefix=None):
    """Load every matching experiment010 KL partition, independently verify."""
    graph_sha = _file_sha256(path)
    cases = []
    for filename in sorted((Path(source_dir) / "cases").glob(
        f"{path.stem}-*-KL-*.json"
    )):
        record = json.loads(filename.read_text(encoding="utf8"))
        settings = record.get("settings", {})
        if (record.get("method") != "KL"
                or record.get("source_method") != source_method
                or settings.get("seed") != source_seed
                or settings.get("kl_passes") != source_passes):
            continue
        if factor_prefix and not record.get("factor_sha256", "").startswith(
            factor_prefix
        ):
            continue
        if record.get("graph_sha256") != graph_sha:
            raise ValueError("Source KL graph SHA256 mismatch")
        cases.append((filename, record))
    if not cases:
        raise FileNotFoundError(
            f"Archived experiment010 KL results absent for {path.stem}: "
            f"{source_dir}/cases. Ensure the user's completed results "
            "are committed and visible after changing branches; "
            "do NOT rerun EVA."
        )
    factors = {r["factor_sha256"] for _, r in cases}
    if len(factors) != 1:
        raise ValueError(
            "Multiple EVA factor variants in KL archive; "
            "pass --factor-sha-prefix to choose exactly one"
        )
    candidates = {}
    provenance = []
    edges = edge_arrays(load_graph(path))
    for filename, record in cases:
        folder = Path(record["archive_path"])
        verified = verify_archive(folder)
        name = record["archive_method"]
        if (verified["status"] != "complete"
                or verified["graph_sha256"] != graph_sha
                or verified["methods"][name]["verified_partitions"]
                != len(record["trials"])):
            raise ValueError(f"Unverified KL source archive: {folder}")
        parts, scores = load_method(folder / f"{name}.npz")
        if len(parts) != len(record["trials"]):
            raise ValueError("KL source restart metadata mismatch")
        provenance.append({
            "case_file": str(filename),
            "case_sha256": _file_sha256(filename),
            "method_npz_sha256": _file_sha256(folder / f"{name}.npz"),
            "archive": str(folder),
        })
        for i, trial in enumerate(record["trials"]):
            restart = int(trial["restart"])
            if restart in candidates:
                raise ValueError(
                    f"Ambiguous/duplicate KL source restart {restart}; "
                    "choose one source seed and KL pass setting."
                )
            score = float(cut_weight(edges, parts[i]))
            if (not np.isclose(score, scores[i], atol=1e-8, rtol=0)
                    or not np.isclose(score, trial["final_cut"],
                                      atol=1e-8, rtol=0)):
                raise ValueError("Original graph KL score mismatch")
            candidates[restart] = (parts[i].astype(np.int8), score)
    return next(iter(factors)), candidates, provenance


def _select_distinct_elites(candidates, top_elites):
    if top_elites < 1:
        raise ValueError("--top-elites must be >= 1")
    ranked = sorted(candidates, key=lambda idx: (-candidates[idx][1], idx))
    selected, seen = [], set()
    for restart in ranked:
        labels, score = candidates[restart]
        # Complement represents exactly the same Max-Cut partition.
        canonical = bytes(np.packbits(
            labels if labels[0] == 0 else 1-labels,
            bitorder="little",
        ))
        if canonical in seen:
            continue
        seen.add(canonical)
        selected.append({
            "restart": int(restart), "labels": labels,
            "initial_cut": float(score),
        })
        if len(selected) == top_elites:
            break
    if not selected:
        raise ValueError("No distinct elite starting partitions")
    return selected


def _init_worker(graph_path, elites):
    global _W_GRAPH, _W_EDGES, _W_ELITES
    _W_GRAPH = load_graph(Path(graph_path))
    _W_EDGES = edge_arrays(_W_GRAPH)
    _W_ELITES = elites


def _config(method, episodes, source_provenance, source_factor,
            graph_sha, seed, budget, source_seed, source_passes,
            source_method, elite_fingerprint):
    settings = {
        "algorithm_version": "experiment011-elite-v1",
        "method": method, "graph_sha256": graph_sha,
        "source_factor_sha256": source_factor,
        "source_method": source_method,
        "source_kl_seed": source_seed,
        "source_kl_passes": source_passes,
        "elite_pool_sha256": elite_fingerprint,
        "source_archive_hashes": sorted(
            r["method_npz_sha256"] for r in source_provenance
        ),
        "episodes": [
            {"source_restart": ep["restart"], "replicate": ep["replicate"],
             "input_cut": ep["initial_cut"]} for ep in episodes
        ],
        "base_seed": seed, "budget": budget,
    }
    key = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf8")
    ).hexdigest()[:20]
    return key, settings


def _verify_case(record, config):
    if record.get("status") != "complete" or record.get("settings") != config:
        raise ValueError("Existing episode chunk settings differ")
    check = verify_archive(record["archive_path"])
    if (check["status"] != "complete"
            or check["graph_sha256"] != config["graph_sha256"]
            or check["methods"][record["archive_method"]][
                "verified_partitions"] != len(record["episodes"])):
        raise ValueError("Existing elite-search archive verification failed")
    _, scores = load_method(
        Path(record["archive_path"]) / f"{record['archive_method']}.npz"
    )
    final = np.array([x["final_cut"] for x in record["episodes"]])
    initial = np.array([x["initial_cut"] for x in record["episodes"]])
    if (not np.allclose(scores, final, rtol=0, atol=1e-8)
            or not np.all(final + 1e-8 >= initial)):
        raise ValueError("Existing elite-search scores or monotonicity invalid")
    return record


def _run_chunk(job):
    method = job["method"]
    config = job["settings"]
    case_path = Path(job["case_path"])
    if case_path.is_file():
        saved = json.loads(case_path.read_text(encoding="utf8"))
        return _verify_case(saved, config), True
    if _W_GRAPH is None or _W_ELITES is None:
        raise RuntimeError("Windows process initializer did not load graph/elites")
    if _file_sha256(job["graph_path"]) != config["graph_sha256"]:
        raise ValueError("Original graph changed during experiment011")
    outcomes, labels_list, cut_list = [], [], []
    for episode in job["episodes"]:
        seed = np.random.SeedSequence([
            job["base_seed"],
            int(Path(job["graph_path"]).stem[1:]),
            int(episode["restart"]), int(episode["replicate"]),
            {"ILS-KL": 1, "EliteMix-KL": 2, "Tabu-KL": 3}[method], 11,
        ])
        rng = np.random.default_rng(seed)
        initial = np.asarray(episode["labels"], dtype=np.int8)
        expected = float(episode["initial_cut"])
        if not np.isclose(
            float(cut_weight(_W_EDGES, initial)), expected, atol=1e-8, rtol=0,
        ):
            raise ValueError("Source elite cut fails original graph rescoring")
        final, detail = optimize_elite(
            _W_GRAPH, initial, _W_ELITES, rng,
            method=method, **job["budget"],
        )
        score = float(cut_weight(_W_EDGES, final))
        if (score + 1e-8 < expected
                or not np.isclose(score, detail["cut_weight"],
                                  atol=1e-8, rtol=0)):
            raise AssertionError("Elite-search final score inconsistent or worse")
        labels_list.append(final)
        cut_list.append(score)
        outcomes.append({
            "restart": int(episode["restart"]),
            "replicate": int(episode["replicate"]),
            "initial_cut": expected,
            "final_cut": score,
            "improvement": score-expected,
            **detail,
        })

    archive_method = f"Elite-{method.replace('-', '')}_EVA"
    trial_path = Path(job["trial_path"])
    archive = new_archive(
        Path(job["output_dir"]) / "archives", new_run_id(),
        Path(job["graph_path"]), _W_GRAPH.shape[0], _W_EDGES,
        2, config, jsonl_output=trial_path,
        argv=["experiment011", method, job["graph_path"]],
    )
    save_method(archive, archive_method, labels_list, cut_list, k=2)
    update_manifest(archive, status="complete", solver_info={
        "method": method, "original_graph_only": True,
        "best_start": max(ep["initial_cut"] for ep in outcomes),
        "best_after": max(cut_list),
        "total_cycles": int(sum(x["cycles_completed"] for x in outcomes)),
        "target": job["budget"]["target"],
    })
    verified = verify_archive(archive["path"])
    if (verified["status"] != "complete"
            or verified["graph_sha256"] != config["graph_sha256"]
            or verified["methods"][archive_method]["verified_partitions"]
            != len(outcomes)):
        raise RuntimeError("Fresh elite-search partition archive failed verification")
    record = {
        "status": "complete", "graph": Path(job["graph_path"]).stem,
        "method": method, "settings": config,
        "graph_sha256": config["graph_sha256"],
        "factor_sha256": config["source_factor_sha256"],
        "archive_method": archive_method,
        "archive_path": str(archive["path"]),
        "trial_path": str(trial_path),
        "episodes": outcomes,
        "best_cut": max(cut_list),
        "source_best": max(x["initial_cut"] for x in outcomes),
    }
    trial_path.parent.mkdir(parents=True, exist_ok=True)
    temp = trial_path.with_suffix(".tmp")
    with temp.open("w", encoding="utf8") as stream:
        for item in outcomes:
            stream.write(json.dumps({
                "graph": record["graph"], "method": method,
                "archive_path": record["archive_path"],
                "graph_sha256": config["graph_sha256"],
                "factor_sha256": config["source_factor_sha256"],
                **item,
            }, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, trial_path)
    _write_json(case_path, record)
    return record, False


def write_summary(dest, *, target):
    dest = Path(dest)
    fields = (
        "graph", "method", "restart", "replicate",
        "initial_cut", "final_cut", "improvement", "cycles_completed",
        "seconds", "termination", "best_events",
        "target", "gap_to_target", "archive_path",
    )
    dest.mkdir(parents=True, exist_ok=True)
    output = dest / "summary.csv"
    temp = output.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for file in sorted((dest / "cases").glob("G*.json")):
            case = json.loads(file.read_text(encoding="utf8"))
            for row in case["episodes"]:
                writer.writerow({
                    "graph": case["graph"], "method": case["method"],
                    **{key: row.get(key) for key in fields},
                    "best_events": len(row["best_events"]),
                    "target": target,
                    "gap_to_target": target - row["final_cut"],
                    "archive_path": case["archive_path"],
                })
    os.replace(temp, output)
    return output


def run_graph(
    graph_path, *, source_dir, dest,
    methods=METHODS, source_method="K-MeansNM-Flip_EVA",
    source_seed=2026, source_passes=30, top_elites=20,
    replicates=4, chunk_size=8, workers=3, base_seed=2026,
    cycles=80, kl_passes=12, kick_sizes=(8,16,32,64,128,256),
    boundary_fraction=0.65, temperature=12.,
    tabu_steps=10000, tabu_tenure=15, tabu_stagnation=3000,
    factor_prefix=None, target=27047.,
):
    graph_path, dest = Path(graph_path), Path(dest)
    if (not methods or len(set(methods)) != len(methods)
            or any(m not in METHODS for m in methods)
            or top_elites < 1 or replicates < 1 or chunk_size < 1
            or workers < 1 or base_seed < 0 or source_seed < 0):
        raise ValueError("Invalid methods, source counts or seed")
    if not kick_sizes or any(k < 1 for k in kick_sizes):
        raise ValueError("Kick sizes must be positive")
    graph_sha = _file_sha256(graph_path)
    factor_sha, candidates, provenance = _source_kl(
        graph_path, source_dir, source_method=source_method,
        source_seed=source_seed, source_passes=source_passes,
        factor_prefix=factor_prefix,
    )
    selected = _select_distinct_elites(candidates, top_elites)
    graph = load_graph(graph_path)
    if max(kick_sizes) > graph.shape[0]:
        raise ValueError("Kick size exceeds graph vertex count")
    elite_pairs = [
        (item["labels"], item["initial_cut"]) for item in selected
    ]
    pool_fingerprint = hashlib.sha256(
        b"".join(
            np.packbits(item["labels"], bitorder="little").tobytes()
            + np.asarray(item["restart"], dtype=np.int64).tobytes()
            for item in selected
        )
    ).hexdigest()
    episodes = []
    for item in selected:
        for replicate in range(replicates):
            episodes.append({
                "restart": item["restart"], "replicate": replicate,
                "initial_cut": item["initial_cut"],
                "labels": item["labels"],
            })
    budget = {
        "cycles": cycles, "kl_passes": kl_passes,
        "kick_sizes": [int(k) for k in kick_sizes],
        "boundary_fraction": boundary_fraction,
        "temperature": temperature,
        "tabu_steps": tabu_steps,
        "tabu_tenure": tabu_tenure,
        "tabu_stagnation": tabu_stagnation,
        "target": target,
    }
    jobs = []
    for first in range(0, len(episodes), chunk_size):
        chunk = episodes[first:first+chunk_size]
        for method in methods:
            fingerprint, settings = _config(
                method, chunk, provenance, factor_sha, graph_sha,
                base_seed, budget, source_seed, source_passes,
                source_method, pool_fingerprint,
            )
            name = f"{graph_path.stem}-{method}-{fingerprint}"
            jobs.append({
                "method": method, "settings": settings,
                "graph_path": str(graph_path),
                "episodes": chunk, "base_seed": base_seed, "budget": budget,
                "output_dir": str(dest),
                "trial_path": str(dest / "trials" / f"{name}.jsonl"),
                "case_path": str(dest / "cases" / f"{name}.json"),
            })
    dest.mkdir(parents=True, exist_ok=True)
    print(
        f"{graph_path.stem}: verified {len(candidates)} KL source cuts; "
        f"selected {len(selected)} distinct elites "
        f"(best {selected[0]['initial_cut']:g}); "
        f"{len(episodes)} episodes/method, {len(jobs)} jobs "
        f"with {workers} independent workers; target={target:g}",
        flush=True,
    )
    records, errors = [], []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker,
        initargs=(str(graph_path), elite_pairs),
    ) as pool:
        futures = {pool.submit(_run_chunk, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result, skipped = future.result()
                records.append(result)
                print(
                    f"{result['graph']} {result['method']} "
                    f"{'verified skip' if skipped else 'complete'} "
                    f"n={len(result['episodes'])} "
                    f"best {result['source_best']:g} -> "
                    f"{result['best_cut']:g} "
                    f"(target gap {target-result['best_cut']:+g})",
                    flush=True,
                )
            except Exception as exc:
                error = {
                    "graph": graph_path.stem,
                    "method": job["method"],
                    "case_path": job["case_path"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc().replace(
                        str(Path.home()), "<HOME>",
                    ),
                }
                errors.append(error)
                _write_json(
                    dest / "errors" /
                    f"{Path(job['case_path']).stem}.json", error,
                )
                print(
                    f"{graph_path.stem} {job['method']} failed: {exc}",
                    flush=True,
                )
            write_summary(dest, target=target)
    if errors:
        raise RuntimeError(
            f"{len(errors)} job(s) failed; saved error details in {dest/'errors'}"
        )
    summary = {
        "graph": graph_path.stem, "source_method": source_method,
        "source_seed": source_seed, "source_passes": source_passes,
        "source_factor_sha256": factor_sha,
        "graph_sha256": graph_sha, "source_cases": provenance,
        "source_KL_best": max(score for _, score in candidates.values()),
        "selected_elite_restarts": [x["restart"] for x in selected],
        "selected_best": selected[0]["initial_cut"],
        "target": target, "budget": budget,
        "replicates": replicates,
        "methods": {},
    }
    for method in methods:
        trials = [
            episode for record in records if record["method"] == method
            for episode in record["episodes"]
        ]
        values = np.array([item["final_cut"] for item in trials])
        starts = np.array([item["initial_cut"] for item in trials])
        best_row = trials[int(np.argmax(values))]
        summary["methods"][method] = {
            "episodes": len(trials), "best": float(values.max()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std(ddof=1)) if len(values)>1 else None,
            "mean_improvement": float((values-starts).mean()),
            "improvement_rate": float(np.mean(values > starts + 1e-8)),
            "best_restart": best_row["restart"],
            "best_replicate": best_row["replicate"],
            "best_gap_to_target": float(target - values.max()),
            "total_seconds": float(sum(t["seconds"] for t in trials)),
        }
    signature = hashlib.sha256(json.dumps({
        "settings": budget, "factor": factor_sha,
        "starts": [(x["restart"], x["initial_cut"]) for x in selected],
        "replicates": replicates, "methods": list(methods),
        "base_seed": base_seed,
    }, sort_keys=True).encode("utf8")).hexdigest()[:16]
    _write_json(
        dest / "summaries" / f"{graph_path.stem}-{signature}.json",
        summary,
    )
    print(
        f"{graph_path.stem}: " + "; ".join(
            f"{method} best={row['best']:g} "
            f"(gap to {target:g}: {row['best_gap_to_target']:+g})"
            for method, row in summary["methods"].items()
        ), flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=["G63"])
    parser.add_argument("--gset-dir", type=Path, default=Path("data"))
    parser.add_argument("--source-dir", type=Path, default=Path(
        "results/gset/experiment010-eva-kl-tabu",
    ))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "results/gset/experiment011-elite-search",
    ))
    parser.add_argument("--source-method", default="K-MeansNM-Flip_EVA",
                        choices=("K-MeansNM-Flip_EVA", "FJ-Flip_EVA"))
    parser.add_argument("--source-kl-seed", type=int, default=2026)
    parser.add_argument("--source-kl-passes", type=int, default=30)
    parser.add_argument("--factor-sha-prefix")
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS))
    parser.add_argument("--top-elites", type=int, default=20)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cycles", type=int, default=80)
    parser.add_argument("--kl-passes", type=int, default=12)
    parser.add_argument("--kick-sizes", nargs="+", type=int,
                        default=[8, 16, 32, 64, 128, 256])
    parser.add_argument("--boundary-fraction", type=float, default=0.65)
    parser.add_argument("--temperature", type=float, default=12.)
    parser.add_argument("--tabu-steps", type=int, default=10000)
    parser.add_argument("--tabu-tenure", type=int, default=15)
    parser.add_argument("--tabu-stagnation", type=int, default=3000)
    parser.add_argument("--target", type=float, default=27047.)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--push-branch")
    args = parser.parse_args()
    if args.push and not args.push_branch:
        parser.error("--push requires --push-branch")
    if args.output_dir.resolve() == args.source_dir.resolve():
        parser.error("Output must differ from KL input directory")
    paths = discover_graphs(args.gset_dir, args.graphs)
    failures = 0
    for graph in paths:
        try:
            run_graph(
                graph, source_dir=args.source_dir, dest=args.output_dir,
                source_method=args.source_method,
                source_seed=args.source_kl_seed,
                source_passes=args.source_kl_passes,
                top_elites=args.top_elites,
                replicates=args.replicates,
                chunk_size=args.chunk_size, workers=args.workers,
                base_seed=args.seed,
                cycles=args.cycles, kl_passes=args.kl_passes,
                kick_sizes=tuple(args.kick_sizes),
                boundary_fraction=args.boundary_fraction,
                temperature=args.temperature,
                tabu_steps=args.tabu_steps,
                tabu_tenure=args.tabu_tenure,
                tabu_stagnation=args.tabu_stagnation,
                target=args.target,
                methods=tuple(args.methods),
                factor_prefix=args.factor_sha_prefix,
            )
            (args.output_dir / "errors" / f"{graph.stem}.json").unlink(
                missing_ok=True,
            )
        except (OSError, KeyError, RuntimeError, ValueError,
                AssertionError, ArithmeticError) as exc:
            failures += 1
            _write_json(args.output_dir / "errors" /
                        f"{graph.stem}.json", {
                            "graph": graph.stem,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc().replace(
                                str(Path.home()), "<HOME>",
                            ),
                        })
            print(f"{graph.stem}: ERROR {exc}", flush=True)
            if not args.continue_on_error:
                if args.push:
                    _git_push(
                        args.output_dir, "experiment011 partial elite runs",
                        allowed_branch=args.push_branch,
                    )
                raise
    if args.push:
        _git_push(
            args.output_dir, "experiment011 KL elite search results",
            allowed_branch=args.push_branch,
        )
    print(
        f"Experiment011: graphs={len(paths)}, errors={failures}, "
        f"summary={args.output_dir / 'summary.csv'}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
