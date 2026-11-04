#!/usr/bin/env python3
"""Experiment026: long-running, resumable Hyperplane Sweep on immutable EVA factors.

No SDP/EVA optimization. One process per graph; append-only complete-batch
traces, SHA-checked best partitions, graph-specific checkpoints and provenance.
The reference file is a comparison target, never an objective or upper bound.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from maxkcut.eva_direct_rounding import optimize_hyperplane
from maxkcut.eva_rounding_cli import _factor_and_case
from maxkcut.experiment023_methods import select_factor, sha, atomic_json
from maxkcut.experiment025_journal import IterationJournal, refine_with_journal
from maxkcut.experiment025_bestcut import algorithm_source_sha256
from maxkcut.io import edge_arrays
from maxkcut.objective import cut_weight

PROTOCOL = "experiment026-hyperplane-semicircle-chunked-v1"
FOCUS = ("G1", "G9", "G28", "G36", "G40", "G42", "G44", "G53")
OUTPUT = Path("results/gset/experiment026-hyperplane-longrun")
REFERENCES = Path("references/gset-literature-2026-11.csv")
REFINEMENT = dict(kl_passes=30, ils_cycles=80, ils_kl_passes=8)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_references(path):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    out = {}
    for row in rows:
        name = row["graph"]
        if name in out or not name.startswith("G"):
            raise ValueError("Invalid duplicate G-set literature reference: " + name)
        cut = int(row["reference_cut"])
        if cut <= 0 or int(row["n_vertices"]) <= 0 or int(row["n_edges"]) <= 0:
            raise ValueError("Invalid graph or cut in reference: " + name)
        bound = row["published_upper_bound"]
        if bound and cut > int(bound):
            raise ValueError("Published feasible cut exceeds its claimed upper bound: " + name)
        out[name] = row
    from maxkcut.experiment023_methods import GRAPH_IDS
    if len(out) != 71 or set(out) != set(GRAPH_IDS):
        raise ValueError("Require literature references for all 71 canonical G-set graphs")
    return out, digest_bytes(path.read_bytes())


def persist_best(folder, batch, labels, edges, cut):
    """Never overwrite a prior winner; independently rescore before and after serialization."""
    labels = np.asarray(labels, dtype=np.int8)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError("Nonbinary winner")
    if abs(float(cut_weight(edges, labels)) - cut) > 1e-8:
        raise AssertionError("Best partition failed original signed-cut rescore")
    path = folder / "partitions" / f"best_batch_{batch:07d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Refusing to overwrite a recorded winner: " + str(path))
    temp = path.with_suffix(".temporary.npz")
    np.savez_compressed(temp, labels=labels)
    os.replace(temp, path)
    with np.load(path, allow_pickle=False) as archived:
        if not np.array_equal(archived["labels"], labels):
            raise AssertionError("Partition changed during save")
    return str(path.relative_to(folder)), sha(path)


def verify_previous_best(folder, state, edges, n):
    rel = state.get("best_partition")
    if rel is None:
        if state["best_cut"] is not None:
            raise ValueError("Best cut has no archived partition")
        return
    path = folder / rel
    if not path.is_file() or sha(path) != state["best_partition_sha256"]:
        raise ValueError("Missing or corrupted saved best cut")
    with np.load(path, allow_pickle=False) as z:
        x = z["labels"]
        if x.shape != (n,) or not np.isin(x, (0, 1)).all():
            raise ValueError("Corrupt best partition labeling")
        if abs(float(cut_weight(edges, x)) - state["best_cut"]) > 1e-8:
            raise ValueError("Best partition does not realize its recorded signed cut")


def run_graph(name, params, ref, refhash):
    """Independent process; only this process writes its own graph directory."""
    from threadpoolctl import threadpool_limits

    root = Path(params["output"])
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "active.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as lockfile:
            lockfile.write(json.dumps(dict(pid=os.getpid(), graph=name, protocol=PROTOCOL)))
        with threadpool_limits(limits=1):
            return _run_locked(name, params, ref, refhash, folder)
    finally:
        lock.unlink(missing_ok=True)


def _run_locked(name, params, ref, refhash, folder):
    source = select_factor(name)
    graph, y, _, graphsha, factorsha = _factor_and_case(Path("data") / (name + ".txt"), source)
    edges = edge_arrays(graph)
    if (graph.shape[0] != int(ref["n_vertices"]) or len(edges[0]) != int(ref["n_edges"])):
        raise ValueError("G-set file topology does not match external literature row: " + name)
    scriptsha = sha(Path(__file__))
    settings = dict(protocol=PROTOCOL, seed=params["seed"],
                    directions_per_batch=params["directions"],
                    batches_per_restart=params["batches_per_restart"],
                    refine_every=params["refine_every"], refinement=REFINEMENT,
                    graph_sha256=graphsha, factor_sha256=factorsha,
                    references_sha256=refhash, script_sha256=scriptsha,
                    optimizer_sha256=algorithm_source_sha256())
    identity = digest_bytes(json.dumps(settings, sort_keys=True).encode("utf-8"))
    marker = folder / "identity.json"
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != dict(settings=settings, identity=identity):
            raise ValueError("Cannot resume graph under altered factor, source, protocol or settings")
    else:
        atomic_json(marker, dict(settings=settings, identity=identity))
    checkpoint = folder / "checkpoint.json"
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state["identity"] != identity:
            raise ValueError("Checkpoint fingerprint mismatch")
    else:
        state = dict(identity=identity, next_batch=0, restart=0, restart_batch=0,
                     normal=None, rng_state=None, best_cut=None, best_partition=None,
                     best_partition_sha256=None, raw_best_cut=None)
    verify_previous_best(folder, state, edges, graph.shape[0])

    stop = time.monotonic() + params["hours"] * 3600
    performed = 0
    reference_cut = int(ref["reference_cut"])
    while time.monotonic() < stop and (
            params["max_batches"] is None or performed < params["max_batches"]):
        batch = state["next_batch"]
        restart, restart_batch = state["restart"], state["restart_batch"]
        if state["normal"] is None:
            rng = np.random.default_rng(np.random.SeedSequence(
                [params["seed"], 26, int(name[1:]), restart]))
            normal = rng.normal(size=y.shape[1])
            normal /= np.linalg.norm(normal)
        else:
            normal = np.asarray(state["normal"], dtype=np.float64)
            if normal.shape != (y.shape[1],) or not np.isfinite(normal).all():
                raise ValueError("Invalid checkpoint hyperplane normal")
            rng = np.random.default_rng()
            rng.bit_generator.state = state["rng_state"]
        trace_path = folder / "traces" / f"batch_{batch:07d}.jsonl"
        report_path = folder / "reports" / f"batch_{batch:07d}.json"
        if trace_path.exists() or report_path.exists():
            raise FileExistsError("Orphaned batch artifact; preserve and audit before resuming: " + str(trace_path))
        began = time.monotonic()
        next_normal, raw_labels, diagnostics = optimize_hyperplane(
            graph, y, rng, initial=normal, directions=params["directions"])
        raw_score = float(cut_weight(edges, raw_labels))
        if abs(raw_score - diagnostics["optimized_raw_cut"]) > 1e-8:
            raise AssertionError("Hyperplane output does not reproduce original signed cut")
        candidate, score, origin = raw_labels, raw_score, "hyperplane"
        # Same established Flip -> KL -> ILS refiner as Experiment025.
        # This is optional and named; the *only* initializer is Hyperplane Sweep.
        if params["refine_every"] and (batch + 1) % params["refine_every"] == 0:
            journal = IterationJournal(folder / "refinements" / f"batch_{batch:07d}",
                                       edges, graph.shape[0], identity)
            candidate, stages, extra, _ = refine_with_journal(
                graph, edges, raw_labels, "hyperplane-sweep", batch,
                int(name[1:]), REFINEMENT, journal)
            score = float(cut_weight(edges, candidate))
            if abs(score - stages["Final"]) > 1e-8:
                raise AssertionError("Refinement best failed rescoring")
            origin = "hyperplane+flip+kl+ils"
        raw_best = max(float(state["raw_best_cut"] or float("-inf")), raw_score)
        best_cut = state["best_cut"]
        is_best = best_cut is None or score > best_cut + 1e-8
        if is_best:
            rel, saved_sha = persist_best(folder, batch, candidate, edges, score)
            best_cut = score
        else:
            rel, saved_sha = state["best_partition"], state["best_partition_sha256"]
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = trace_path.with_suffix(".temporary.jsonl")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for item in diagnostics["trace"]:
                stream.write(json.dumps(dict(graph=name, restart=restart, batch=batch, **item),
                                        sort_keys=True) + "\n")
        os.replace(temporary, trace_path)
        atomic_json(report_path, dict(graph=name, batch=batch, restart=restart,
                restart_batch=restart_batch, raw_score=raw_score, refined_score=score,
                candidate_origin=origin, best_cut=best_cut,
                reference_cut=reference_cut, gap_to_reference=reference_cut-best_cut,
                event_count=len(diagnostics["trace"]), accepted=diagnostics["improvements"],
                runtime_seconds=time.monotonic()-began, winning_partition=rel if is_best else None))
        continued = restart_batch + 1 < params["batches_per_restart"]
        next_state = dict(identity=identity, next_batch=batch+1,
            restart=restart if continued else restart+1,
            restart_batch=restart_batch+1 if continued else 0,
            normal=next_normal.tolist() if continued else None,
            rng_state=rng.bit_generator.state if continued else None,
            best_cut=best_cut, best_partition=rel, best_partition_sha256=saved_sha,
            raw_best_cut=raw_best)
        # The state pointer is written LAST. Fail closed on interrupted batches.
        atomic_json(checkpoint, next_state)
        state = next_state
        performed += 1
        print(json.dumps(dict(graph=name,batch=batch+1,raw=raw_score,
                              best=best_cut,reference=reference_cut,
                              gap=reference_cut-best_cut,
                              refinement=origin)), flush=True)
    return dict(graph=name, batches_completed=state["next_batch"],
                best_cut=state["best_cut"], reference_cut=reference_cut,
                gap_to_reference=reference_cut-state["best_cut"] if state["best_cut"] is not None else None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", nargs="+", default=list(FOCUS))
    ap.add_argument("--hours", type=float, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--directions", type=int, default=60)
    ap.add_argument("--batches-per-restart", type=int, default=10)
    ap.add_argument("--refine-every", type=int, default=5,
                    help="0 disables Flip/KL/ILS; otherwise refine every N completed batches")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="Safe short pilot or extension; limited per graph per invocation")
    ap.add_argument("--references", type=Path, default=REFERENCES)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if (not args.graphs or len(set(args.graphs)) != len(args.graphs)
            or args.hours <= 0 or not 1 <= args.workers <= 32
            or args.directions < 3 or args.directions % 3
            or args.batches_per_restart < 1 or args.refine_every < 0
            or (args.max_batches is not None and args.max_batches < 1)):
        ap.error("Invalid graphs, hours, worker count, batch length or restart/refinement options")
    if Path(args.output).resolve() == Path("results/gset/experiment025-gw16-bestcut-quality").resolve():
        ap.error("Never write Experiment026 outputs into Experiment025")
    references, refsha = load_references(args.references)
    if any(g not in references for g in args.graphs):
        ap.error("One or more selected graphs are not in the 71-graph literature table")
    params = dict(output=str(args.output), hours=args.hours, seed=args.seed,
                  directions=args.directions, batches_per_restart=args.batches_per_restart,
                  refine_every=args.refine_every, max_batches=args.max_batches)
    plan = dict(protocol=PROTOCOL, graphs=args.graphs, references_sha256=refsha,
                workers=min(args.workers, len(args.graphs)), params=params,
                targets={g:int(references[g]["reference_cut"]) for g in args.graphs})
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return 0
    # Graphs run independently with one writer per graph; no Git operations or SDP solves.
    results=[]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(args.graphs))) as pool:
        jobs={pool.submit(run_graph, g, params, references[g], refsha):g for g in args.graphs}
        for future in as_completed(jobs):
            results.append(future.result())
    args.output.mkdir(parents=True, exist_ok=True)
    summary = args.output / "latest_summary.csv"
    fields = ["graph","batches_completed","best_cut","reference_cut","gap_to_reference"]
    temporary = summary.with_suffix(".temporary.csv")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda x:int(x["graph"][1:])))
    os.replace(temporary, summary)
    print(json.dumps(dict(completed_graphs=len(results), summary=str(summary)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
