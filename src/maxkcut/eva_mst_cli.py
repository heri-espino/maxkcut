"""Experiment005: one exact ambient-space MST longest-edge cut per metric.

Consumes an already completed EVA factor. NEVER reruns the SDP or EVA.
One deterministic cut for each of L1, L2 and Linf. Saves exact MST
parents/edge distances, the binary partition and independently verified
weighted Max-Cut scores, plus a SHA256-preserved factor snapshot.
Optional --push commits once AFTER the entire batch (no per-edge pushes).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from .archive import (
    _atomic_npz, load_method, new_archive, new_run_id, save_method,
    update_manifest, verify_archive,
)
from .eva_cli import _git_push, _write_json
from .eva_mst import METRICS, METHODS, _cut_longest_tree_edge, mst_binary_partition
from .eva_rounding_cli import _factor_and_case, _immutable_factor_copy
from .io import edge_arrays
from .objective import cut_weight
from .sweep import discover_graphs


def _case_path(root, graph_name, factor_sha, metric):
    return root / "cases" / f"{graph_name}-{factor_sha[:16]}-mst-{metric}.json"


def _verify_existing(record, *, graph_name, graph_sha, factor_sha, metric):
    """A completed case is skipped only if both graph/tree/partition verify."""
    if (record.get("status") != "complete"
            or record.get("graph") != graph_name
            or record.get("graph_sha256") != graph_sha
            or record.get("factor_sha256") != factor_sha
            or record.get("metric") != metric
            or record.get("method") != METHODS[metric]):
        raise ValueError("Existing MST record has different graph/factor/settings")
    archive = Path(record["archive_path"])
    verification = verify_archive(archive)
    if (verification["status"] != "complete"
            or verification["graph_sha256"] != graph_sha
            or verification["methods"][METHODS[metric]]["verified_partitions"] != 1):
        raise ValueError("Existing MST partition archive is invalid")
    labels, scores = load_method(archive / f"{METHODS[metric]}.npz")
    with np.load(record["tree_path"], allow_pickle=False) as tree:
        parent = tree["parent"].copy()
        lengths = tree["edge_lengths"].copy()
    reconstructed, removed_child, sizes = _cut_longest_tree_edge(parent, lengths)
    if (len(labels) != 1 or not np.array_equal(reconstructed, labels[0])
            or abs(float(scores[0]) - float(record["cut_weight"])) > 1e-8
            or removed_child != record["removed_edge_child"]
            or list(sizes) != record["component_sizes"]):
        raise ValueError("Stored MST/tree partition integrity check failed")
    return record


def run_graph(path, *, source, dest, metrics):
    """Compute or verify each metric independently, with an immutable factor."""
    source, dest = Path(source), Path(dest)
    graph, vectors, eva_case, graph_sha, factor_sha = _factor_and_case(path, source)
    dest.mkdir(parents=True, exist_ok=True)
    factor_snapshot = _immutable_factor_copy(source, dest, path.stem, factor_sha)
    edges = edge_arrays(graph)
    results = []
    for metric in metrics:
        method = METHODS[metric]
        case_path = _case_path(dest, path.stem, factor_sha, metric)
        if case_path.is_file():
            saved = json.loads(case_path.read_text(encoding="utf8"))
            checked = _verify_existing(
                saved, graph_name=path.stem, graph_sha=graph_sha,
                factor_sha=factor_sha, metric=metric,
            )
            print(f"{path.stem}/{method}: verified skip; cut={checked['cut_weight']}",
                  flush=True)
            results.append(checked)
            continue

        print(
            f"{path.stem}/{method}: exact MST of {len(vectors)} vectors in "
            f"R^{vectors.shape[1]} (no PCA; O(n^2*p) work)",
            flush=True,
        )
        labels, tree_report, parent, distances = mst_binary_partition(vectors, metric)
        score = float(cut_weight(edges, labels))
        tree_path = (dest / "trees" /
                     f"{path.stem}-{factor_sha[:16]}-mst-{metric}.npz")
        _atomic_npz(
            tree_path,
            parent=parent, edge_lengths=distances,
            removed_edge_child=np.asarray(
                tree_report["removed_edge_child"], dtype=np.int64
            ),
        )
        # One archive per graph/metric is immutable. A later failed method
        # does not invalidate any previous completed method.
        trial_path = dest / "trials" / (
            f"{path.stem}-{factor_sha[:16]}-mst-{metric}.jsonl"
        )
        settings = {
            "algorithm": "exact complete-graph Prim MST, remove longest edge",
            "metric": metric, "method": method,
            "source": "saved full-dimensional EVA factor; no PCA",
            "graph_sha256": graph_sha, "factor_sha256": factor_sha,
            "factor_rank": int(vectors.shape[1]), "factor_sweeps":
                eva_case["factor_sweeps"],
            "random_restarts": 0, "deterministic_partitions": 1,
        }
        archive = new_archive(
            dest / "archives", new_run_id(), path, len(vectors), edges,
            2, settings, jsonl_output=trial_path, argv=sys.argv,
        )
        archive["manifest"]["graph_source"] = str(path)
        archive["manifest"]["results_jsonl"] = str(trial_path)
        update_manifest(archive)
        save_method(archive, method, [labels], [score], k=2)
        update_manifest(archive, status="complete", solver_info={
            "solver": "no optimizer; exact ambient MST postprocessing",
            **tree_report,
        })
        check = verify_archive(archive["path"])
        if (check["status"] != "complete"
                or check["methods"][method]["verified_partitions"] != 1
                or abs(check["methods"][method]["best_cut"] - score) > 1e-8):
            raise RuntimeError("Saved MST cut did not independently verify")
        trial_path.parent.mkdir(parents=True, exist_ok=True)
        temp_trial = trial_path.with_suffix(".jsonl.tmp")
        temp_trial.write_text(json.dumps({
            "graph": path.stem, "method": method,
            "metric": metric, "cut_weight": score,
            "factor_sha256": factor_sha,
            "graph_sha256": graph_sha,
            "archive_path": str(archive["path"]),
            **tree_report,
        }, sort_keys=True) + "\n", encoding="utf8")
        os.replace(temp_trial, trial_path)
        record = {
            "graph": path.stem, "vertices": int(len(vectors)),
            "status": "complete",
            "metric": metric, "method": method,
            "cut_weight": score,
            "graph_sha256": graph_sha, "factor_sha256": factor_sha,
            "factor_rank": int(vectors.shape[1]),
            "factor_sweeps": int(eva_case["factor_sweeps"]),
            "factor_snapshot": str(factor_snapshot),
            "archive_path": str(archive["path"]),
            "tree_path": str(tree_path),
            "trial_path": str(trial_path),
            "base_eva_random_rounding_best_20": {
                name: stats["best"]
                for name, stats in eva_case["methods"].items()
            },
            "baseline_caution":
                "MST gives one deterministic candidate; baselines show best "
                "of 20 randomized trials, not matched sampling budgets.",
            **tree_report,
        }
        _write_json(case_path, record)
        results.append(record)
        print(
            f"{path.stem}/{method}: weighted cut={score:g}; "
            f"component sizes={tree_report['component_sizes']}; "
            f"tree time={tree_report['seconds_excluding_jit_compilation']:.3f}s",
            flush=True,
        )
    return results


def write_summary(dest):
    dest = Path(dest)
    output = dest / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "graph", "metric", "method", "vertices", "factor_rank",
        "factor_sweeps", "cut_weight", "largest_edge_length",
        "mst_total_length", "component_size_0", "component_size_1",
        "seconds_excluding_jit_compilation", "graph_sha256",
        "factor_sha256", "archive_path", "tree_path",
    ]
    tmp = output.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for file in sorted((dest / "cases").glob("G*-mst-*.json")):
            case = json.loads(file.read_text(encoding="utf8"))
            if case.get("status") != "complete":
                continue
            writer.writerow({
                **{key: case.get(key) for key in fields},
                "component_size_0": case["component_sizes"][0],
                "component_size_1": case["component_sizes"][1],
            })
    os.replace(tmp, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=["G11"])
    parser.add_argument("--gset-dir", type=Path, default=Path("data"))
    parser.add_argument("--source-dir", type=Path,
                        default=Path("results/gset/experiment003-eva-numba"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/gset/experiment005-eva-mst"))
    parser.add_argument("--metrics", nargs="+", choices=METRICS,
                        default=list(METRICS))
    parser.add_argument("--max-n", type=int, default=2000,
                        help="Explicitly opt into n=20000; exact MST is O(n^2*p)")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--push", action="store_true",
                        help="Commit and push results ONCE, after ALL graphs")
    parser.add_argument("--push-branch",
                        help="Required exact branch for --push")
    args = parser.parse_args()
    if not args.metrics or len(set(args.metrics)) != len(args.metrics):
        parser.error("Specify each metric at most once")
    if args.max_n < 2:
        parser.error("--max-n must be >= 2")
    if args.push and not args.push_branch:
        parser.error("--push requires --push-branch")
    if args.output_dir.resolve() == args.source_dir.resolve():
        parser.error("Output directory must differ from saved factor directory")
    paths = discover_graphs(args.gset_dir, args.graphs)
    failures = 0
    for path in paths:
        # Fast .txt header preflight avoids loading large factors by mistake.
        if path.suffix.lower() == ".txt":
            with path.open("r", encoding="utf8") as file:
                n = int(file.readline().split()[0])
            if n > args.max_n:
                parser.error(
                    f"{path.stem}: n={n} > --max-n={args.max_n}; "
                    "increase the cap explicitly after cost review"
                )
    for path in paths:
        try:
            run_graph(
                path, source=args.source_dir, dest=args.output_dir,
                metrics=args.metrics,
            )
            (args.output_dir / "errors" / f"{path.stem}.json").unlink(
                missing_ok=True
            )
        except (OSError, ValueError, RuntimeError, ImportError,
                AssertionError, ArithmeticError) as exc:
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
                        args.output_dir, f"experiment005 partial {path.stem} MST",
                        allowed_branch=args.push_branch,
                    )
                raise
        write_summary(args.output_dir)
    if args.push:
        _git_push(
            args.output_dir, "experiment005: exact original-space EVA MST cuts",
            allowed_branch=args.push_branch,
        )
    print(
        f"Experiment005: {len(paths)} graphs, {failures} failures. "
        f"Summary: {args.output_dir / 'summary.csv'}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
