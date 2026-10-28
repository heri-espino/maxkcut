"""Experiment007: optimize the original weighted Max-Cut again after MST-BestCut.

Stage 1: reuse the *archived* partition obtained by selecting the best
single-edge split of an exact full-dimensional EVA MST (L1/L2/Linf).
Stage 2: run signed-weight positive-gain vertex flips to local optimality
(or a declared maximum sweep count).

Neither EVA nor an MST is recomputed. A new cut never scores below its
verified saved MST starting cut. Every partition is independently archived.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from .archive import (
    load_method, new_archive, new_run_id, save_method, update_manifest,
    verify_archive,
)
from .eva_cli import _git_push, _write_json
from .eva_mst import METRICS
from .eva_mst_center_cli import _method, _verify_central
from .io import edge_arrays, load_graph
from .local_search import improve_by_flips
from .objective import cut_weight
from .sweep import discover_graphs


def _find_bestcut_record(mst_dir, graph_name, metric, sha_prefix=None):
    matches = []
    for file in sorted(
        (Path(mst_dir) / "center-cases").glob(
            f"{graph_name}-*-mst-{metric}-bestcut.json"
        )
    ):
        record = json.loads(file.read_text(encoding="utf8"))
        if sha_prefix and not record.get("factor_sha256", "").startswith(sha_prefix):
            continue
        matches.append((file, record))
    if not matches:
        raise FileNotFoundError(
            f"No archived MST-BestCut for {graph_name}/{metric} in {mst_dir}. "
            "Run maxkcut.eva_mst_center_cli with --rules bestcut first."
        )
    if len(matches) != 1:
        raise ValueError(
            f"{graph_name}/{metric}: {len(matches)} saved factor versions; "
            "pass --factor-sha-prefix to choose exactly one."
        )
    return matches[0]


def _verify_refinement(record, graph_name, graph_sha, factor_sha,
                       metric, seed, max_sweeps, source_cut):
    method = f"MST-BestCut-Flip-{metric.upper()}_EVA"
    if (record.get("status") != "complete"
            or record.get("graph") != graph_name
            or record.get("graph_sha256") != graph_sha
            or record.get("factor_sha256") != factor_sha
            or record.get("metric") != metric
            or record.get("method") != method
            or record.get("flip_seed") != seed
            or record.get("max_flip_sweeps") != max_sweeps
            or not np.isclose(record.get("initial_cut"), source_cut,
                              atol=1e-8, rtol=0)):
        raise ValueError("Existing MST local refinement uses different inputs")
    verified = verify_archive(Path(record["archive_path"]))
    if (verified["status"] != "complete"
            or verified["graph_sha256"] != graph_sha
            or verified["methods"][method]["verified_partitions"] != 1
            or not np.isclose(verified["methods"][method]["best_cut"],
                              record["refined_cut"], atol=1e-8, rtol=0)):
        raise ValueError("Saved MST refinement archive failed verification")
    return record


def refine_saved_bestcut(path, *, mst_dir, dest, metrics=METRICS,
                         flip_seed=2026, max_flip_sweeps=20,
                         factor_sha_prefix=None):
    """Run or verify the second optimization from each metric's saved e* cut."""
    path, mst_dir, dest = Path(path), Path(mst_dir), Path(dest)
    if max_flip_sweeps < 1 or flip_seed < 0:
        raise ValueError("max_flip_sweeps >= 1 and flip_seed >= 0 are required")
    if len(set(metrics)) != len(metrics) or any(metric not in METRICS for metric in metrics):
        raise ValueError("Duplicate or unsupported distance metric")
    graph = load_graph(path)
    edges = edge_arrays(graph)
    graph_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    dest.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for metric in metrics:
        source_path, source = _find_bestcut_record(
            mst_dir, path.stem, metric, factor_sha_prefix,
        )
        factor_sha = source["factor_sha256"]
        _verify_central(
            source, name=path.stem, graph_sha=graph_sha,
            factor_sha=factor_sha, rule="bestcut", metric=metric,
        )
        original_method = _method(metric, "bestcut")
        initial_partitions, initial_scores = load_method(
            Path(source["archive_path"]) / f"{original_method}.npz"
        )
        if initial_partitions.shape != (1, graph.shape[0]):
            raise ValueError("Exactly one saved MST-BestCut partition is required")
        start_labels = np.asarray(initial_partitions[0], dtype=np.int8)
        initial_score = float(initial_scores[0])
        if not np.isclose(
            float(cut_weight(edges, start_labels)), initial_score,
            atol=1e-8, rtol=0,
        ):
            raise RuntimeError("Original MST-BestCut score failed direct rescoring")
        method = f"MST-BestCut-Flip-{metric.upper()}_EVA"
        output_case = dest / "cases" / (
            f"{path.stem}-{factor_sha[:16]}-mst-{metric}"
            f"-flip{max_flip_sweeps}-seed{flip_seed}.json"
        )
        if output_case.is_file():
            saved = json.loads(output_case.read_text(encoding="utf8"))
            saved = _verify_refinement(
                saved, path.stem, graph_sha, factor_sha,
                metric, flip_seed, max_flip_sweeps, initial_score,
            )
            print(f"{path.stem}/{method}: verified skip, "
                  f"{initial_score:g} -> {saved['refined_cut']:g}", flush=True)
            outcomes.append(saved)
            continue

        rng = np.random.default_rng(np.random.SeedSequence([
            flip_seed, int(path.stem[1:]), {"l1": 1, "l2": 2, "linf": 3}[metric], 7,
        ]))
        refined_labels, detail = improve_by_flips(
            graph, start_labels.copy(), rng, max_sweeps=max_flip_sweeps,
        )
        refined_score = float(cut_weight(edges, refined_labels))
        if not np.isclose(detail["initial_cut"], initial_score,
                          atol=1e-8, rtol=0):
            raise RuntimeError("Local search input did not equal archived MST cut")
        if refined_score < initial_score - 1e-8:
            raise AssertionError("A strictly improving flip stage reduced Max-Cut")

        output_trial = dest / "trials" / (
            f"{path.stem}-{factor_sha[:16]}-mst-{metric}"
            f"-flip{max_flip_sweeps}-seed{flip_seed}.jsonl"
        )
        settings = {
            "algorithm": "MST-BestCut (all tree edges) + positive-gain flips",
            "optimizer_stage": "stage 2, binary local refinement only",
            "stage1_archive": source["archive_path"],
            "stage1_record": str(source_path),
            "stage1_cut_weight": initial_score,
            "metric": metric,
            "max_flip_sweeps": max_flip_sweeps,
            "flip_seed": flip_seed,
            "graph_sha256": graph_sha,
            "factor_sha256": factor_sha,
            "source_factor_rank": source["factor_rank"],
            "full_ambient_embedding": True,
            "reuse_saved_factor_and_tree": True,
            "random_restarts": 0,
            "local_search_order": "deterministic given seed; shuffled vertex sweeps",
        }
        archive = new_archive(
            dest / "archives", new_run_id(), path, graph.shape[0],
            edges, 2, settings, jsonl_output=output_trial, argv=sys.argv,
        )
        save_method(archive, method, [refined_labels], [refined_score], k=2)
        update_manifest(archive, status="complete", solver_info={
            "solver": "existing signed-weight greedy vertex-flip local search",
            "original_mst_cut": initial_score,
            "refined_cut": refined_score,
            "improvement": refined_score - initial_score,
            "flip_sweeps_limit": max_flip_sweeps,
            **detail,
        })
        checked = verify_archive(archive["path"])
        if (checked["status"] != "complete"
                or checked["methods"][method]["verified_partitions"] != 1
                or not np.isclose(checked["methods"][method]["best_cut"],
                                  refined_score, atol=1e-8, rtol=0)):
            raise RuntimeError("Refined partition failed independent archive verification")
        record = {
            "status": "complete",
            "graph": path.stem,
            "vertices": graph.shape[0],
            "metric": metric,
            "method": method,
            "initial_cut": initial_score,
            "refined_cut": refined_score,
            "improvement": refined_score - initial_score,
            "flip_count": int(detail["flips"]),
            "flip_sweeps": int(detail["sweeps"]),
            "termination": detail["termination"],
            "local_seconds": float(detail["seconds"]),
            "max_flip_sweeps": max_flip_sweeps,
            "flip_seed": flip_seed,
            "source_mst_bestcut_case": str(source_path),
            "source_mst_bestcut_archive": source["archive_path"],
            "original_mst_removed_edge_child": source["removed_edge_child"],
            "source_factor_rank": source["factor_rank"],
            "factor_sha256": factor_sha,
            "graph_sha256": graph_sha,
            "archive_path": str(archive["path"]),
            "trial_path": str(output_trial),
        }
        output_trial.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_trial.with_suffix(".jsonl.tmp")
        tmp.write_text(json.dumps(record, sort_keys=True) + "\n",
                       encoding="utf8")
        os.replace(tmp, output_trial)
        _write_json(output_case, record)
        outcomes.append(record)
        print(
            f"{path.stem}/{method}: {initial_score:g} -> {refined_score:g} "
            f"(+{refined_score-initial_score:g}), flips={detail['flips']}, "
            f"sweeps={detail['sweeps']}, {detail['termination']}",
            flush=True,
        )
    return outcomes


def write_summary(dest):
    dest = Path(dest)
    path = dest / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "graph", "metric", "method", "initial_cut", "refined_cut",
        "improvement", "flip_count", "flip_sweeps", "termination",
        "local_seconds", "max_flip_sweeps", "flip_seed",
        "source_factor_rank", "graph_sha256", "factor_sha256",
        "source_mst_bestcut_archive", "archive_path",
    )
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record_path in sorted((dest / "cases").glob("G*.json")):
            record = json.loads(record_path.read_text(encoding="utf8"))
            if record.get("status") == "complete":
                writer.writerow({name: record.get(name) for name in fields})
    os.replace(tmp, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=["G63"])
    parser.add_argument("--gset-dir", type=Path, default=Path("data"))
    parser.add_argument("--mst-dir", type=Path,
                        default=Path("results/gset/experiment005-eva-mst"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/gset/experiment007-eva-mst-flips"))
    parser.add_argument("--metrics", nargs="+", choices=METRICS,
                        default=list(METRICS))
    parser.add_argument("--flip-sweeps", type=int, default=20)
    parser.add_argument("--flip-seed", type=int, default=2026)
    parser.add_argument("--factor-sha-prefix",
                        help="Disambiguate multiple archived factor revisions")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--push", action="store_true",
                        help="Push all results ONCE after finishing the batch")
    parser.add_argument("--push-branch")
    args = parser.parse_args()
    if args.push and not args.push_branch:
        parser.error("--push requires --push-branch")
    if args.output_dir.resolve() == args.mst_dir.resolve():
        parser.error("Output must differ from saved original MST directory")
    if args.flip_sweeps < 1 or args.flip_seed < 0:
        parser.error("Invalid flip sweep count or seed")
    paths = discover_graphs(args.gset_dir, args.graphs)
    failures = 0
    for path in paths:
        try:
            refine_saved_bestcut(
                path, mst_dir=args.mst_dir, dest=args.output_dir,
                metrics=args.metrics, flip_seed=args.flip_seed,
                max_flip_sweeps=args.flip_sweeps,
                factor_sha_prefix=args.factor_sha_prefix,
            )
            (args.output_dir / "errors" / f"{path.stem}.json").unlink(missing_ok=True)
        except (OSError, ValueError, RuntimeError, AssertionError,
                ArithmeticError, KeyError) as exc:
            failures += 1
            _write_json(args.output_dir / "errors" / f"{path.stem}.json", {
                "graph": path.stem,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().replace(
                    str(Path.home()), "<HOME>"
                ),
            })
            print(f"{path.stem}: FAILED {type(exc).__name__}: {exc}", flush=True)
            if not args.continue_on_error:
                write_summary(args.output_dir)
                if args.push:
                    _git_push(
                        args.output_dir, "experiment007: partial MST flip results",
                        allowed_branch=args.push_branch,
                    )
                raise
        write_summary(args.output_dir)
    if args.push:
        _git_push(
            args.output_dir, "experiment007: MST-BestCut plus local-flip results",
            allowed_branch=args.push_branch,
        )
    print(f"Experiment007: {len(paths)} graphs; failures={failures}; "
          f"summary={args.output_dir / 'summary.csv'}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
