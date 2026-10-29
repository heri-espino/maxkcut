"""Experiment009: prune q longest saved EVA MST edges; optimize *binary* Max-Cut.

NO EVA runs, PCA projection or new geometric MST. Consume exact archived
full-ambient MSTs from experiment005/006 and original signed G-set graphs.
For each q and L1/L2/Linf: q+1 components -> contracted signed graph ->
coarse component-label local search -> unrestricted vertex-flip refinement.

Each q has an independently verified, resumable archive of ALL coarse
trial partitions plus the refined final partition. --push is ONCE at end.
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
from .eva_mst_cli import _verify_existing
from .eva_mst_multicut import multi_edge_candidates
from .io import edge_arrays, load_graph
from .sweep import discover_graphs


def _select_tree(mst_dir, graph_name, metric, factor_sha_prefix):
    matches = []
    for path in sorted((Path(mst_dir) / "cases").glob(
        f"{graph_name}-*-mst-{metric}.json"
    )):
        record = json.loads(path.read_text(encoding="utf8"))
        if factor_sha_prefix is None or record.get(
            "factor_sha256", "",
        ).startswith(factor_sha_prefix):
            matches.append(record)
    if not matches:
        raise FileNotFoundError(
            f"No saved MST for {graph_name}/{metric}. Run experiment005 first."
        )
    if len(matches) != 1:
        raise ValueError(
            f"{graph_name}/{metric}: {len(matches)} MST factors; "
            "provide --factor-sha-prefix to select one."
        )
    return matches[0]


def _fingerprint(graph_sha, factor_sha, metric, q, seed, restarts,
                 coarse_sweeps, vertex_sweeps, tree_sha, q_prefix):
    settings = {
        "graph_sha256": graph_sha,
        "factor_sha256": factor_sha,
        "tree_sha256": tree_sha,
        "metric": metric, "q": q, "seed": seed,
        "nested_q_prefix": list(q_prefix),
        "restarts": restarts,
        "coarse_sweeps": coarse_sweeps,
        "vertex_sweeps": vertex_sweeps,
        "algorithm_version": "nested-fullspace-mst-multicut-v1",
    }
    token = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf8")
    ).hexdigest()[:16]
    return token, settings


def _verify_saved(record, settings):
    if record.get("status") != "complete" or record.get("settings") != settings:
        raise ValueError("Saved multi-edge experiment identity/settings mismatch")
    report = verify_archive(Path(record["archive_path"]))
    methods = record["method_names"]
    if (report["status"] != "complete"
            or report["graph_sha256"] != settings["graph_sha256"]
            or report["methods"][methods[0]]["verified_partitions"]
            != len(record["coarse_trials"])
            or report["methods"][methods[1]]["verified_partitions"] != 1):
        raise ValueError("Saved multi-edge partition archive is incomplete")
    _, initial_scores = load_method(
        Path(record["archive_path"]) / f"{methods[0]}.npz"
    )
    _, final_scores = load_method(
        Path(record["archive_path"]) / f"{methods[1]}.npz"
    )
    if (not np.allclose(
            initial_scores, [x["coarse_cut"] for x in record["coarse_trials"]],
            atol=1e-7, rtol=0)
            or not np.isclose(final_scores[0], record["final_cut"],
                              atol=1e-7, rtol=0)):
        raise ValueError("Saved multi-edge scores mismatch individual trials")
    return record


def run_graph(path, *, mst_dir, dest, metrics=METRICS,
              q_values=(1, 2, 4, 8, 16, 32, 64, 128),
              coarse_restarts=20, coarse_sweeps=20, vertex_sweeps=20,
              seed=2026, factor_sha_prefix=None):
    """Reuse verified exact MSTs, archive q-level runs as they complete."""
    path, dest, mst_dir = Path(path), Path(dest), Path(mst_dir)
    graph = load_graph(path)
    n = graph.shape[0]
    graph_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    edges = edge_arrays(graph)
    dest.mkdir(parents=True, exist_ok=True)
    if not q_values or max(q_values) >= n:
        raise ValueError(f"q must be <= n-1={n-1} for {path.stem}")
    all_results = []
    for metric in metrics:
        base = _select_tree(mst_dir, path.stem, metric, factor_sha_prefix)
        factor_sha = base["factor_sha256"]
        _verify_existing(
            base, graph_name=path.stem, graph_sha=graph_sha,
            factor_sha=factor_sha, metric=metric,
        )
        tree_path = Path(base["tree_path"])
        tree_sha = hashlib.sha256(tree_path.read_bytes()).hexdigest()
        with np.load(tree_path, allow_pickle=False) as saved:
            parent = saved["parent"].copy()
            lengths = saved["edge_lengths"].copy()
        # The generator reuses the same saved tree for every q and
        # monotonically carries forward the previous coarse best.
        generator = multi_edge_candidates(
            graph, parent, lengths, q_values,
            coarse_restarts=coarse_restarts,
            coarse_sweeps=coarse_sweeps,
            flip_sweeps=vertex_sweeps,
            seed=seed, graph_number=int(path.stem[1:]),
            metric_index={"l1": 1, "l2": 2, "linf": 3}[metric],
        )
        for data in generator:
            q = data["q"]
            token, settings = _fingerprint(
                graph_sha, factor_sha, metric, q, seed, coarse_restarts,
                coarse_sweeps, vertex_sweeps, tree_sha,
                q_values[:list(q_values).index(q) + 1],
            )
            case_path = dest / "cases" / f"{path.stem}-mst-{metric}-q{q}-{token}.json"
            if case_path.is_file():
                record = json.loads(case_path.read_text(encoding="utf8"))
                all_results.append(_verify_saved(record, settings))
                print(f"{path.stem} MST-{metric} q={q}: verified skip "
                      f"final={record['final_cut']:g}", flush=True)
                continue

            label = {"l1": "L1", "l2": "L2", "linf": "Linf"}[metric]
            coarse_method = f"MST-Multi-{label}-q{q}-Coarse_EVA"
            final_method = f"MST-Multi-{label}-q{q}-Flip_EVA"
            trial_path = dest / "trials" / f"{path.stem}-mst-{metric}-q{q}-{token}.jsonl"
            archive = new_archive(
                dest / "archives", new_run_id(), path, n, edges,
                2, settings, jsonl_output=trial_path, argv=sys.argv,
            )
            save_method(
                archive, coarse_method, data["coarse_labels"],
                data["coarse_scores"], k=2,
            )
            save_method(
                archive, final_method, [data["final_labels"]],
                [data["final_cut"]], k=2,
            )
            update_manifest(archive, status="complete", solver_info={
                "solver": "contracted signed Max-Cut + positive-gain vertex flips",
                "geometric_mst_reused": True,
                "pca_projection": False,
                "components": q + 1,
                "coarse_best_cut": data["best_coarse_cut"],
                "final_cut": data["final_cut"],
                "flip_termination": data["vertex_termination"],
            })
            checked = verify_archive(archive["path"])
            if (checked["status"] != "complete"
                    or checked["graph_sha256"] != graph_sha
                    or checked["methods"][coarse_method]["verified_partitions"]
                    != len(data["coarse_labels"])
                    or checked["methods"][final_method]["verified_partitions"]
                    != 1):
                raise RuntimeError("Experiment009 partition archive did not verify")
            record = {
                "status": "complete", "graph": path.stem, "metric": metric,
                "q": q, "components": q + 1, "vertices": n,
                "graph_sha256": graph_sha, "factor_sha256": factor_sha,
                "factor_rank": base["factor_rank"],
                "tree_path": str(tree_path),
                "settings": settings, "method_names": [
                    coarse_method, final_method,
                ],
                "removed_edge_children": data["removed_edge_children"],
                "removed_edge_lengths": data["removed_edge_lengths"],
                "component_sizes": data["component_sizes"],
                "coarse_trials": data["coarse_trials"],
                "best_coarse_trial_index": data["best_coarse_trial_index"],
                "best_coarse_cut": data["best_coarse_cut"],
                "final_cut": data["final_cut"],
                "improvement": data["vertex_improvement"],
                "vertex_flips": data["vertex_flips"],
                "vertex_sweeps": data["vertex_sweeps"],
                "vertex_termination": data["vertex_termination"],
                "vertex_seconds": data["vertex_seconds"],
                "longest_edge_baseline": base["cut_weight"],
                "archive_path": str(archive["path"]),
                "trial_path": str(trial_path),
                "note": (
                    "q removed MST edges make q+1 forest components, "
                    "but the Max-Cut assigns those components to TWO sides."
                ),
            }
            trial_path.parent.mkdir(parents=True, exist_ok=True)
            temp_trial = trial_path.with_suffix(".jsonl.tmp")
            with temp_trial.open("w", encoding="utf8") as stream:
                for trial in data["coarse_trials"]:
                    stream.write(json.dumps({
                        "graph": path.stem, "metric": metric, "q": q,
                        "graph_sha256": graph_sha,
                        "factor_sha256": factor_sha,
                        "archive_path": str(archive["path"]),
                        **trial,
                    }, sort_keys=True) + "\n")
                stream.write(json.dumps({
                    "graph": path.stem, "metric": metric, "q": q,
                    "kind": "vertex_flip_refinement",
                    "cut_weight": data["final_cut"],
                    "initial_coarse_cut": data["best_coarse_cut"],
                    "flips": data["vertex_flips"],
                    "sweeps": data["vertex_sweeps"],
                    "termination": data["vertex_termination"],
                }, sort_keys=True) + "\n")
            os.replace(temp_trial, trial_path)
            _write_json(case_path, record)
            all_results.append(record)
            print(
                f"{path.stem} MST-{metric} q={q}: "
                f"{len(data['coarse_trials'])} coarse trials, "
                f"best={data['best_coarse_cut']:g}, "
                f"after vertex flips={data['final_cut']:g} "
                f"(+{data['vertex_improvement']:g})",
                flush=True,
            )
    return all_results


def write_summary(dest):
    dest = Path(dest)
    output = dest / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "graph", "metric", "q", "components", "vertices", "factor_rank",
        "best_coarse_cut", "final_cut", "improvement",
        "longest_edge_baseline", "best_coarse_trial_index",
        "vertex_flips", "vertex_sweeps", "vertex_termination",
        "vertex_seconds", "graph_sha256", "factor_sha256", "archive_path",
    )
    temp = output.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for path in sorted((dest / "cases").glob("G*-mst-*.json")):
            record = json.loads(path.read_text(encoding="utf8"))
            if record.get("status") == "complete":
                writer.writerow({key: record.get(key) for key in fields})
    os.replace(temp, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=["G63"])
    parser.add_argument("--gset-dir", type=Path, default=Path("data"))
    parser.add_argument("--mst-dir", type=Path,
                        default=Path("results/gset/experiment005-eva-mst"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/gset/experiment009-mst-multiedge"))
    parser.add_argument("--metrics", nargs="+", choices=METRICS,
                        default=list(METRICS))
    parser.add_argument("--q-values", nargs="+", type=int,
                        default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--coarse-restarts", type=int, default=20)
    parser.add_argument("--coarse-sweeps", type=int, default=20)
    parser.add_argument("--vertex-sweeps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--factor-sha-prefix")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--push", action="store_true",
                        help="Push only ONCE at end of the batch")
    parser.add_argument("--push-branch")
    args = parser.parse_args()
    if args.push and not args.push_branch:
        parser.error("--push requires --push-branch")
    if args.output_dir.resolve() == args.mst_dir.resolve():
        parser.error("--output-dir must differ from --mst-dir")
    if (len(set(args.metrics)) != len(args.metrics)
            or not args.q_values
            or args.q_values[0] != 1
            or args.q_values != sorted(set(args.q_values))
            or args.seed < 0
            or args.coarse_restarts < 0
            or args.coarse_sweeps < 1
            or args.vertex_sweeps < 1):
        parser.error("Invalid metrics, q schedule, seed or sweep settings")
    paths = discover_graphs(args.gset_dir, args.graphs)
    errors = 0
    for path in paths:
        try:
            run_graph(
                path, mst_dir=args.mst_dir, dest=args.output_dir,
                metrics=args.metrics, q_values=args.q_values,
                coarse_restarts=args.coarse_restarts,
                coarse_sweeps=args.coarse_sweeps,
                vertex_sweeps=args.vertex_sweeps,
                seed=args.seed, factor_sha_prefix=args.factor_sha_prefix,
            )
            (args.output_dir / "errors" / f"{path.stem}.json").unlink(missing_ok=True)
        except (OSError, ValueError, KeyError, RuntimeError,
                AssertionError, ArithmeticError) as exc:
            errors += 1
            _write_json(args.output_dir / "errors" / f"{path.stem}.json", {
                "graph": path.stem,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().replace(
                    str(Path.home()), "<HOME>"
                ),
            })
            print(f"{path.stem}: ERROR {type(exc).__name__}: {exc}", flush=True)
            if not args.continue_on_error:
                write_summary(args.output_dir)
                if args.push:
                    _git_push(
                        args.output_dir, "experiment009 partial MST multi-edge",
                        allowed_branch=args.push_branch,
                    )
                raise
        write_summary(args.output_dir)
    if args.push:
        _git_push(
            args.output_dir, "experiment009: MST multi-edge signed Max-Cut results",
            allowed_branch=args.push_branch,
        )
    print(
        f"Experiment009: {len(paths)} graphs, errors={errors}, "
        f"results={args.output_dir / 'summary.csv'}", flush=True,
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
