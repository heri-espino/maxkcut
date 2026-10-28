"""Experiment006: several principled ways to choose ONE edge from an EVA MST.

Reuse (or compute once) each existing exact L1/L2/Linf MST over the
original full-dimensional EVA vectors. Evaluate balanced, diameter
center and direct best weighted Max-Cut edge selections. Repeated MST
optimization, PCA and randomized restarts are deliberately absent.
All output is archived, verified, resumable and pushed once at the end.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from .archive import (
    _atomic_npz, load_method, new_archive, new_run_id, save_method, update_manifest,
    verify_archive,
)
from .eva_cli import _git_push, _write_json
from .eva_mst import METRICS, METHODS
from .eva_mst_center import (
    METHOD_PREFIX, RULES, balanced_edge, best_cut_edge,
    diameter_center_edge, labels_cutting_edge,
)
from .eva_mst_cli import run_graph as run_longest_mst, _verify_existing
from .eva_mst_interpret import characterize_best_edge
from .eva_rounding_cli import _factor_and_case
from .io import edge_arrays
from .objective import cut_weight
from .sweep import discover_graphs


def _method(metric, rule):
    metric_suffix = {"l1": "L1", "l2": "L2", "linf": "Linf"}[metric]
    return f"{METHOD_PREFIX[rule]}-{metric_suffix}_EVA"


def _verify_central(record, *, name, graph_sha, factor_sha, rule, metric):
    if (record.get("status") != "complete"
            or record.get("graph") != name
            or record.get("graph_sha256") != graph_sha
            or record.get("factor_sha256") != factor_sha
            or record.get("rule") != rule
            or record.get("metric") != metric
            or record.get("method") != _method(metric, rule)):
        raise ValueError("Central MST case identity mismatch")
    archive_path = Path(record["archive_path"])
    verified = verify_archive(archive_path)
    if (verified["status"] != "complete"
            or verified["graph_sha256"] != graph_sha
            or verified["methods"][_method(metric, rule)]["verified_partitions"]
            != 1):
        raise ValueError("Central MST archived partition cannot be verified")
    labels, scores = load_method(archive_path / f"{_method(metric, rule)}.npz")
    if not np.isclose(float(scores[0]), record["cut_weight"], rtol=0, atol=1e-8):
        raise ValueError("Central MST score differs from partition archive")
    with np.load(record["tree_path"], allow_pickle=False) as stored:
        parent = stored["parent"].copy()
        lengths = stored["edge_lengths"].copy()
    reconstructed = labels_cutting_edge(parent, record["removed_edge_child"])
    if not np.array_equal(reconstructed, labels[0]):
        raise ValueError("Saved central MST cut does not match removed tree edge")
    return record


def run_central_graph(path, *, source, dest, metrics, rules):
    """Reuse the same exact MST for all alternative single-edge rules."""
    source, dest = Path(source), Path(dest)
    graph, vectors, eva_case, graph_sha, factor_sha = _factor_and_case(path, source)
    # Every original MST archive is verified (not merely assumed complete);
    # a missing one is computed exactly once, with NO EVA rerun.
    baseline = run_longest_mst(
        path, source=source, dest=dest, metrics=metrics,
    )
    by_metric = {case["metric"]: case for case in baseline}
    edges = edge_arrays(graph)
    results = []
    for metric in metrics:
        baseline_case = by_metric[metric]
        with np.load(baseline_case["tree_path"], allow_pickle=False) as npz:
            parent = npz["parent"].copy()
            distances = npz["edge_lengths"].copy()
        # Interpret the highest-score edge against EVERY edge on the same
        # exact MST; preserve full edge profiles for independent research.
        interpretation, edge_profiles = characterize_best_edge(
            parent, distances, edges,
        )
        profile_base = f"{path.stem}-{factor_sha[:16]}-mst-{metric}"
        profile_npz = dest / "edge-profiles" / f"{profile_base}.npz"
        if profile_npz.is_file():
            with np.load(profile_npz, allow_pickle=False) as saved_profile:
                if (set(saved_profile.files) != set(edge_profiles)
                        or any(not np.array_equal(saved_profile[key], val)
                               for key, val in edge_profiles.items())):
                    raise ValueError("Existing MST edge profile does not match factor/tree")
        else:
            _atomic_npz(profile_npz, **edge_profiles)
        _write_json(dest / "edge-profile-reports" / f"{profile_base}.json", {
            "graph": path.stem, "metric": metric,
            "graph_sha256": graph_sha, "factor_sha256": factor_sha,
            "tree_path": baseline_case["tree_path"],
            "edge_profile_path": str(profile_npz),
            **interpretation,
        })
        for rule in rules:
            method = _method(metric, rule)
            case_path = (dest / "center-cases" /
                         f"{path.stem}-{factor_sha[:16]}-mst-{metric}-{rule}.json")
            if case_path.is_file():
                saved = json.loads(case_path.read_text(encoding="utf8"))
                verified = _verify_central(
                    saved, name=path.stem, graph_sha=graph_sha,
                    factor_sha=factor_sha, rule=rule, metric=metric,
                )
                print(
                    f"{path.stem}/{method}: verified skip cut={verified['cut_weight']}",
                    flush=True,
                )
                results.append(verified)
                continue
            if rule == "balanced":
                removed, details = balanced_edge(parent, distances)
            elif rule == "diameter_center":
                removed, details = diameter_center_edge(parent, distances)
            elif rule == "bestcut":
                removed, details = best_cut_edge(parent, distances, edges)
            else:
                raise ValueError(f"Invalid edge selection rule: {rule}")
            labels = labels_cutting_edge(parent, removed)
            score = float(cut_weight(edges, labels))
            if rule == "bestcut" and not np.isclose(
                score, details["best_tree_edge_cut_weight"],
                rtol=1e-10, atol=1e-7,
            ):
                raise ArithmeticError(
                    "Path-sum cut weight disagrees with direct weighted-graph scoring"
                )
            trial_path = (dest / "center-trials" /
                          f"{path.stem}-{factor_sha[:16]}-mst-{metric}-{rule}.jsonl")
            settings = {
                "algorithm": "one-edge split of an exact original-ambient EVA MST",
                "metric": metric, "rule": rule, "method": method,
                "graph_sha256": graph_sha, "factor_sha256": factor_sha,
                "factor_rank": int(vectors.shape[1]),
                "source_mst_tree": baseline_case["tree_path"],
                "source_mst_archive": baseline_case["archive_path"],
                "random_restarts": 0,
                "exact_deterministic_cuts": 1,
            }
            archive = new_archive(
                dest / "center-archives", new_run_id(), path,
                len(vectors), edges, 2, settings,
                jsonl_output=trial_path, argv=sys.argv,
            )
            save_method(archive, method, [labels], [score], k=2)
            update_manifest(archive, status="complete", solver_info={
                "solver": "no EVA, no SDP; deterministic MST postprocessing",
                "removed_edge_child": removed,
                "removed_edge_parent": int(parent[removed]),
                "component_sizes": [int((labels == 0).sum()),
                                    int((labels == 1).sum())],
                **details,
            })
            checked = verify_archive(archive["path"])
            if (checked["status"] != "complete"
                    or checked["methods"][method]["verified_partitions"] != 1):
                raise RuntimeError("Central MST partition archive failed verification")
            trial = {
                "graph": path.stem, "rule": rule, "metric": metric,
                "method": method, "cut_weight": score,
                "removed_edge_child": removed,
                "graph_sha256": graph_sha, "factor_sha256": factor_sha,
                "archive_path": str(archive["path"]), **details,
            }
            trial_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = trial_path.with_suffix(".jsonl.tmp")
            tmp.write_text(json.dumps(trial, sort_keys=True) + "\n", encoding="utf8")
            os.replace(tmp, trial_path)
            case = {
                **trial, "status": "complete", "vertices": int(len(vectors)),
                "factor_rank": int(vectors.shape[1]),
                "factor_sweeps": int(eva_case["factor_sweeps"]),
                "tree_path": baseline_case["tree_path"],
                "original_longest_edge_cut": baseline_case["cut_weight"],
                "component_sizes": [int((labels == 0).sum()),
                                    int((labels == 1).sum())],
                "removed_edge_parent": int(parent[removed]),
                "archive_path": str(archive["path"]),
                "source_eva_dir": str(source),
                **({"best_edge_interpretation": interpretation,
                    "edge_profile_path": str(profile_npz)}
                   if rule == "bestcut" else {}),
            }
            _write_json(case_path, case)
            results.append(case)
            print(
                f"{path.stem}/{method}: cut={score:g}, "
                f"sizes={case['component_sizes']}, "
                f"longest-edge baseline={baseline_case['cut_weight']:g}",
                flush=True,
            )
    return results


def write_center_summary(dest):
    dest = Path(dest)
    target = dest / "center-summary.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    names = (
        "graph", "metric", "rule", "method", "vertices", "factor_rank",
        "cut_weight", "original_longest_edge_cut", "removed_edge_parent",
        "removed_edge_child", "component_size_0", "component_size_1",
        "graph_sha256", "factor_sha256", "archive_path", "tree_path",
    )
    tmp = target.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for file in sorted((dest / "center-cases").glob("G*.json")):
            case = json.loads(file.read_text(encoding="utf8"))
            writer.writerow({
                **{key: case.get(key) for key in names},
                "component_size_0": case["component_sizes"][0],
                "component_size_1": case["component_sizes"][1],
            })
    os.replace(tmp, target)
    profile_summary = dest / "edge-profile-summary.csv"
    fields = (
        "graph", "metric", "best_edge_child", "best_edge_parent",
        "best_edge_cut_score", "best_edge_length", "best_edge_balance_pairs",
        "component_imbalance", "best_edge_is_longest",
        "best_edge_is_most_balanced",
        "best_edge_is_diameter_center_choice",
        "best_edge_length_percentile", "best_edge_balance_percentile",
        "spearman_cut_vs_length", "spearman_cut_vs_balance",
        "positive_cross_weight", "negative_cross_weight",
        "existing_cross_edge_count",
        "mean_signed_weight_per_potential_cross_pair",
        "graph_sha256", "factor_sha256", "edge_profile_path",
    )
    tmp = profile_summary.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for file in sorted((dest / "edge-profile-reports").glob("G*.json")):
            row = json.loads(file.read_text(encoding="utf8"))
            writer.writerow({key: row.get(key) for key in fields})
    os.replace(tmp, profile_summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=["G11"])
    parser.add_argument("--gset-dir", type=Path, default=Path("data"))
    parser.add_argument("--source-dir", type=Path,
                        default=Path("results/gset/experiment003-eva-numba"))
    parser.add_argument("--fallback-source-dir", nargs="*", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/gset/experiment005-eva-mst"))
    parser.add_argument("--metrics", nargs="+", choices=METRICS,
                        default=list(METRICS))
    parser.add_argument("--rules", nargs="+", choices=RULES,
                        default=list(RULES))
    parser.add_argument("--max-n", type=int, default=2000,
                        help="Exact MST needs O(n^2 p) work; opt in to >2000")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--push", action="store_true",
                        help="Commit and push once after all graphs")
    parser.add_argument("--push-branch")
    args = parser.parse_args()
    if args.push and not args.push_branch:
        parser.error("--push needs --push-branch")
    if len(set(args.rules)) != len(args.rules):
        parser.error("Duplicate edge rules")
    if len(set(args.metrics)) != len(args.metrics):
        parser.error("Duplicate distance metrics")
    paths = discover_graphs(args.gset_dir, args.graphs)
    for path in paths:
        if path.suffix.lower() == ".txt":
            with path.open("r", encoding="utf8") as stream:
                n = int(stream.readline().split()[0])
            if n > args.max_n:
                parser.error(
                    f"{path.stem}: {n} vertices exceeds --max-n={args.max_n}"
                )
    failures = 0
    for path in paths:
        try:
            candidates = [args.source_dir] + args.fallback_source_dir
            source = next(
                (folder for folder in candidates
                 if (folder / "cases" / f"{path.stem}.json").is_file()
                 and (folder / "factors" / f"{path.stem}.npz").is_file()),
                None,
            )
            if source is None:
                raise FileNotFoundError(
                    f"No saved EVA factor for {path.stem} in {candidates}"
                )
            run_central_graph(
                path, source=source, dest=args.output_dir,
                metrics=args.metrics, rules=args.rules,
            )
            (args.output_dir / "center-errors" /
             f"{path.stem}.json").unlink(missing_ok=True)
        except (OSError, ValueError, RuntimeError, ArithmeticError,
                ImportError, AssertionError, KeyError) as exc:
            failures += 1
            _write_json(args.output_dir / "center-errors" /
                        f"{path.stem}.json", {
                            "graph": path.stem,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc().replace(
                                str(Path.home()), "<HOME>"
                            ),
                        })
            print(f"{path.stem}: FAILED {type(exc).__name__}: {exc}", flush=True)
            if not args.continue_on_error:
                write_center_summary(args.output_dir)
                if args.push:
                    _git_push(
                        args.output_dir, "experiment006 partial center-edge results",
                        allowed_branch=args.push_branch,
                    )
                raise
        write_center_summary(args.output_dir)
    if args.push:
        _git_push(
            args.output_dir, "experiment006: central MST edge choices",
            allowed_branch=args.push_branch,
        )
    print(
        f"Experiment006: {len(paths)} graphs, failures={failures}, "
        f"results={args.output_dir / 'center-summary.csv'}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
