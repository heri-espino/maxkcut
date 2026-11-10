"""Experiment027: angular-guided greedy Max-Cut rounding over immutable EVA factors.

Only NEW methods run. The 16 GW normals, signed original graph, frozen
Experiment025 Flip/KL/ILS and its per-iteration journal are reused. This file
lives outside src/maxkcut/ to preserve Experiment025's source fingerprint.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

# Set before numerical imports when no explicit user environment exists.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
from scipy import sparse

from maxkcut.eva_direct_rounding import optimize_semicircle
from maxkcut.eva_rounding_cli import _factor_and_case
from maxkcut.experiment023_methods import (
    GRAPH_IDS, select_factor, write_case, verify_case, atomic_json, sha,
)
from maxkcut.experiment024_paired_gw16 import directions_archive
from maxkcut.experiment025_bestcut import SETTINGS, algorithm_source_sha256
from maxkcut.experiment025_journal import IterationJournal, refine_with_journal
from maxkcut.io import edge_arrays
from maxkcut.objective import cut_weight

METHODS = ("angular-margin-greedy", "angular-gain-greedy", "angular-pair-greedy")
ROOT = Path("results/gset/experiment027-angular-greedy")
OLD_ROOT = Path("results/gset/experiment025-gw16-bestcut-quality")
PROTOCOL = "experiment027-gw16-angular-greedy-v1"
FROZEN_025_SOURCE_SHA = "99c0b15ea1309f1a0931e3baba6735f79a1eb6622567d8f27e66be13a898f8ad"


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / norm if np.isfinite(norm) and norm > 1e-12 else None


def angular_margins(vectors, normal):
    """Geodesic distance to each vertex's crossing boundary, in radians."""
    norm = np.linalg.norm(vectors, axis=1)
    if np.any(norm <= 1e-12):
        raise ValueError("Zero embedding row has undefined angular margin")
    cos = np.clip((vectors @ normal) / norm, -1.0, 1.0)
    return np.arcsin(np.clip(np.abs(cos), 0.0, 1.0))


def _tangent_to_vertex(vertex, normal):
    c = float(vertex @ normal)
    lateral = vertex - c * normal
    orient = -1.0 if c >= 0 else 1.0
    return _unit(orient * lateral)


def _tangent_to_pair(v1, v2, normal):
    """Toward the nearest unit normal satisfying two crossing constraints."""
    a = np.column_stack((v1, v2))
    if np.linalg.matrix_rank(a, tol=1e-9) < 2:
        return None
    projection = normal - a @ (np.linalg.pinv(a.T @ a, rcond=1e-12) @ (a.T @ normal))
    target = _unit(projection)
    if target is None:
        return None
    if target @ normal < 0:
        target = -target
    return _unit(target - float(target @ normal) * normal)


def _choices(graph, vectors, normal, method, count):
    n = vectors.shape[0]
    labels = (vectors @ normal > 0).astype(np.int8)
    spins = 1.0 - 2.0 * labels.astype(np.float64)
    gains = spins * np.asarray(graph @ spins).ravel()
    margins = angular_margins(vectors, normal)
    # A larger positive flip gain and smaller angular distance merit priority.
    score = np.maximum(0.0, gains) / (margins + 1e-3)
    if method == "angular-margin-greedy":
        order = np.lexsort((np.arange(n), margins))
    else:
        order = np.lexsort((np.arange(n), margins, -score))
    options = []
    if method == "angular-pair-greedy":
        pool = order[:min(n, max(2 * count, 8))]
        pairs = sorted(itertools.combinations(pool, 2),
                       key=lambda ab: (-(score[ab[0]] + score[ab[1]]),
                                       margins[ab[0]] + margins[ab[1]],
                                       min(ab), max(ab)))
        for i, j in pairs:
            tangent = _tangent_to_pair(vectors[i], vectors[j], normal)
            if tangent is not None:
                options.append(((int(i), int(j)), tangent))
            if len(options) >= count:
                break
    else:
        for i in order:
            tangent = _tangent_to_vertex(vectors[i], normal)
            if tangent is not None:
                options.append(((int(i),), tangent))
            if len(options) >= count:
                break
    return options, gains, margins


def greedy_angular(graph, vectors, initial, method, *, rounds=32,
                   candidates=8, on_iteration=None):
    """Best-of-K exact semicircle paths at each greedy iteration.

    The search is exact only on each tested great semicircle; no global or
    combinatorial local-optimality claim is made. No new EVA solve occurs.
    """
    if method not in METHODS or rounds < 1 or candidates < 1:
        raise ValueError("Invalid angular method or search budget")
    y = np.asarray(vectors, dtype=np.float64)
    graph = sparse.csr_matrix(graph)
    r = _unit(initial)
    if (r is None or y.ndim != 2 or y.shape[1] != r.size
            or graph.shape != (len(y), len(y)) or not np.isfinite(y).all()):
        raise ValueError("Incompatible graph, EVA factor, or initial normal")
    edges = edge_arrays(graph)
    best = (y @ r > 0).astype(np.int8)
    best_score = float(cut_weight(edges, best))
    initial_score = best_score
    trace = []
    accepted = [dict(iteration=0, normal=r.tolist(), cut=best_score)]
    if on_iteration is not None:
        on_iteration(dict(stage="GW", iteration=0, current_cut=best_score), best.copy())
    stop = "round_limit"
    for iteration in range(1, rounds + 1):
        choices, gains, margins = _choices(graph, y, r, method, candidates)
        if not choices:
            stop = "no_valid_tangent"
            break
        next_best = best_score
        selected = None
        attempts = []
        for rank, (vertices, tangent) in enumerate(choices, 1):
            candidate_r, labels, info = optimize_semicircle(graph, y, r, tangent)
            score = float(cut_weight(edges, labels))
            if not np.isclose(score, info["realized_arc_cut"], atol=1e-8, rtol=0):
                raise AssertionError("Exact signed cut differs from angular sweep")
            if score > next_best + 1e-8:
                next_best, selected = score, (candidate_r.copy(), labels.copy(), vertices)
            record = dict(stage="AngularCandidate", iteration=iteration,
                          candidate=rank, vertices=list(vertices),
                          angular_margins=[float(margins[i]) for i in vertices],
                          flip_gains=[float(gains[i]) for i in vertices],
                          theta=float(info["theta"]),
                          numerically_valid=bool(info["numerically_valid"]),
                          events=int(info["events"]),
                          current_cut=score, incumbent_cut=best_score)
            attempts.append(record)
            if on_iteration is not None:
                on_iteration(record, labels.copy())
        trace.extend(attempts)
        if selected is None:
            stop = "no_strict_improvement_among_candidates"
            break
        r, best, which = selected
        best_score = next_best
        accepted.append(dict(iteration=iteration, vertices=list(which),
                             normal=r.tolist(), cut=best_score))
        if on_iteration is not None:
            on_iteration(dict(stage="AngularAccepted", iteration=iteration,
                              vertices=list(which), current_cut=best_score),best.copy())
    if not np.array_equal((y @ r > 0).astype(np.int8), best):
        raise AssertionError("Saved best labels do not match the accepted normal")
    if float(cut_weight(edges, best)) != best_score:
        raise AssertionError("Saved best labels were not scored on original graph")
    return r, best, dict(initial_raw_cut=initial_score,
                         optimized_raw_cut=best_score,
                         paths_evaluated=len(trace),
                         accepted_iterations=len(accepted)-1,
                         termination=stop, trace=trace,
                         accepted_normals=accepted)


def _source_identity(graph_sha, factor_sha, directions_sha, method, phase, config):
    payload = dict(protocol=PROTOCOL,graph_sha=graph_sha,factor_sha=factor_sha,
                   directions_sha=directions_sha,method=method,phase=phase,
                   settings=config,experiment025_source_sha=algorithm_source_sha256(),
                   runner_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _old_baseline(baseline_root, phase, graph, graph_sha, factor_sha, direction_sha):
    folder = Path(baseline_root)/phase/graph
    shared = folder/"gw16_common.npz"
    status = folder/"gw-rr"/"status.json"
    if not shared.is_file() or not status.is_file():
        return None
    with np.load(shared, allow_pickle=False) as f:
        old_normals = f["normals"]
    old_status = json.loads(status.read_text(encoding="utf8"))
    if old_status.get("status") != "complete_16_best_seen":
        return None  # A partial pilot is not a verified historical comparator.
    if (old_status.get("graph_sha256") != graph_sha
            or old_status.get("factor_sha256") != factor_sha
            or old_status.get("shared_directions_sha256") != direction_sha):
        raise ValueError("Experiment025 baseline differs in graph/factor/GW directions")
    hyper_path = folder/"hyperplane-sweep"/"status.json"
    hyper_best = None
    hyper_sha = None
    if hyper_path.is_file():
        hyper_status=json.loads(hyper_path.read_text(encoding="utf8"))
        if hyper_status.get("status") == "complete_16_best_seen":
            if (hyper_status.get("graph_sha256") != graph_sha
                    or hyper_status.get("factor_sha256") != factor_sha
                    or hyper_status.get("shared_directions_sha256") != direction_sha):
                raise ValueError("Experiment025 Hyperplane Sweep has mismatched provenance")
            hyper_best=float(hyper_status["best_final"])
            hyper_sha=sha(hyper_path)
    return dict(gw_refined_best=float(old_status["best_final"]),
                hyperplane_refined_best=hyper_best,
                old_status_sha256=sha(status),
                hyperplane_status_sha256=hyper_sha,
                normals=old_normals)


def run_one(graph, method, phase, args):
    if graph not in GRAPH_IDS or method not in METHODS or phase not in SETTINGS:
        raise ValueError("Unknown graph, method, or phase")
    if algorithm_source_sha256() != FROZEN_025_SOURCE_SHA:
        raise RuntimeError("Experiment025 frozen optimizer source fingerprint changed")
    if args.factor_dir:
        factor_dir = Path(args.factor_dir)
    else:
        factor_dir = select_factor(graph)
    source = Path(args.data_dir)/f"{graph}.txt"
    g, y, _, graph_sha, factor_sha = _factor_and_case(source, factor_dir)
    edges = edge_arrays(g)
    folder = Path(args.root)/phase/graph
    normals, gw_labels, direction_meta = directions_archive(
        folder, y, graph, graph_sha, factor_sha)
    directions_sha = direction_meta["direction_rows_sha256"]
    old = _old_baseline(args.baseline_root, phase, graph,
                        graph_sha, factor_sha, directions_sha)
    if old is not None and not np.array_equal(old["normals"], normals):
        raise ValueError("Saved Experiment025 and Experiment027 GW normals differ")
    cfg = dict(SETTINGS[phase], rounds=int(args.rounds),
               candidates=int(args.candidates), objective="original_signed_maxcut")
    ident = _source_identity(graph_sha,factor_sha,directions_sha,method,phase,cfg)
    target = folder/method
    target.mkdir(parents=True, exist_ok=True)
    details = []
    chosen = range(16) if args.seed is None else [args.seed]
    for index in chosen:
        saved = verify_case(folder, method, index, g, edges, ident)
        if saved is None:
            journal_path = target/"progress"/f"{index:04d}"
            # Partial journals are not silently replayed or overwritten.
            if journal_path.exists() and any(journal_path.iterdir()):
                raise RuntimeError(f"Incomplete seed journal; audit before retry: {journal_path}")
            journal = IterationJournal(journal_path, edges, len(y), ident)
            started = time.monotonic()
            r, raw, extra = greedy_angular(
                g, y, normals[index], method,
                rounds=args.rounds, candidates=args.candidates,
                on_iteration=journal.observe)
            best_raw, best_raw_cut = journal.assert_best()
            if not np.array_equal(best_raw, raw):
                raise AssertionError("Best initializer journal differs from selected raw cut")
            raw_cut = float(cut_weight(edges, raw))
            if not np.isclose(raw_cut,best_raw_cut,atol=1e-8,rtol=0):
                raise AssertionError("Initializer cut differs from journal")
            # Store normal trajectory in independent, checksum-verifiable archive.
            accept_file = journal_path/"accepted_normals.npz"
            temp = journal_path/"accepted_normals.temporary.npz"
            np.savez_compressed(temp,
                normals=np.asarray([a["normal"] for a in extra["accepted_normals"]]),
                cuts=np.asarray([a["cut"] for a in extra["accepted_normals"]]),
                iterations=np.asarray([a["iteration"] for a in extra["accepted_normals"]]))
            os.replace(temp,accept_file)
            accepts_sha=sha(accept_file)
            final, stages, refinements, partitions = refine_with_journal(
                g, edges, raw, method, index, int(graph[1:]), cfg, journal)
            if not np.isclose(stages["Round"],raw_cut,atol=1e-8,rtol=0):
                raise AssertionError("Refinement used a different raw partition")
            metadata = dict(experiment="experiment027",graph=graph,method=method,
                phase=phase,index=index,fingerprint=ident,
                graph_sha256=graph_sha,factor_sha256=factor_sha,
                shared_direction_sha256=directions_sha,
                gw_start_cut=float(cut_weight(edges,gw_labels[index])),
                raw_cut=raw_cut, final_cut=stages["Final"],
                best_seen_cut=journal.best_cut,
                last_iteration_cut=refinements["ils_trace"][-1]["candidate_cut"]
                    if refinements["ils_trace"] else stages["KL"],
                best_seen_event=journal.best_event,progress_events=journal.events,
                progress_improvements=journal.improvements,
                accepted_normals_sha256=accepts_sha,
                angular_summary={k:v for k,v in extra.items()
                                 if k not in ("trace","accepted_normals")},
                refinement_details=refinements,
                seed_seconds=time.monotonic()-started,
                gw_refined_best_exp025=old["gw_refined_best"] if old else None,
                baseline_status_sha256=old["old_status_sha256"] if old else None,
                hyperplane_refined_best_exp025=old["hyperplane_refined_best"] if old else None,
                hyperplane_status_sha256=old["hyperplane_status_sha256"] if old else None)
            write_case(folder,method,index,final,stages,metadata,partitions)
            saved=verify_case(folder,method,index,g,edges,ident)
            if saved is None:
                raise AssertionError("New seed checkpoint failed verification")
        details.append(saved)
    # Global status includes all VERIFIED seed archives, not only this invocation.
    all_saved = [verify_case(folder,method,i,g,edges,ident) for i in range(16)]
    completed = [s for s in all_saved if s is not None]
    row=dict(experiment="experiment027",graph=graph,method=method,phase=phase,
        fingerprint=ident,graph_sha256=graph_sha,factor_sha256=factor_sha,
        shared_directions_sha256=directions_sha,
        starts_completed=len(completed),starts_requested=16,
        status="complete_16" if len(completed)==16 else "partial",
        best_final=max(s["best_seen_cut"] for s in completed),
        mean_final=float(np.mean([s["best_seen_cut"] for s in completed])),
        best_raw=max(s["raw_cut"] for s in completed),
        raw_mean=float(np.mean([s["raw_cut"] for s in completed])),
        gw_refined_best_exp025=old["gw_refined_best"] if old else None,
        baseline_status_sha256=old["old_status_sha256"] if old else None,
        hyperplane_refined_best_exp025=old["hyperplane_refined_best"] if old else None,
        hyperplane_status_sha256=old["hyperplane_status_sha256"] if old else None,
        total_paths=sum(s["angular_summary"]["paths_evaluated"] for s in completed),
        selected_seed_index=max(completed,key=lambda s:s["best_seen_cut"])["index"])
    atomic_json(target/"status.json",row)
    return row


def run_graph(graph, args):
    return [run_one(graph,m,args.phase,args)
            for m in (METHODS if args.method=="all" else [args.method])]


def summary(args):
    import csv
    rows=[]
    for graph in GRAPH_IDS:
        for method in METHODS:
            fn=Path(args.root)/args.phase/graph/method/"status.json"
            if fn.exists():
                state=json.loads(fn.read_text(encoding="utf8"))
                rows.append({k:state.get(k) for k in (
                    "graph","method","phase","status","starts_completed",
                    "best_final","mean_final","best_raw","raw_mean",
                    "gw_refined_best_exp025","hyperplane_refined_best_exp025",
                    "total_paths")})
    output=Path(args.root)/"analysis"/f"{args.phase}_method_summary.csv"
    output.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        with output.open("w",newline="",encoding="utf8") as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    return dict(rows=len(rows),expected=71*len(METHODS),path=str(output),
                fully_completed=sum(row["status"]=="complete_16" for row in rows))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("run","summary"),default="run")
    parser.add_argument("--graphs",nargs="+",choices=GRAPH_IDS,default=["G11"])
    parser.add_argument("--method",choices=("all",)+METHODS,default="all")
    parser.add_argument("--phase",choices=("pilot","full"),default="pilot")
    parser.add_argument("--rounds",type=int,default=None)
    parser.add_argument("--candidates",type=int,default=None)
    parser.add_argument("--seed",type=int,choices=range(16),default=None)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--baseline-root",type=Path,default=OLD_ROOT)
    parser.add_argument("--data-dir",type=Path,default=Path("data"))
    parser.add_argument("--factor-dir",type=Path,default=None)
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv)
    if args.rounds is None:args.rounds=2 if args.phase=="pilot" else 32
    if args.candidates is None:args.candidates=2 if args.phase=="pilot" else 8
    if args.workers<1 or args.rounds<1 or args.candidates<1:
        parser.error("workers, rounds, and candidates must be positive")
    if args.action=="summary":
        print(json.dumps(summary(args),indent=2));return
    plan=dict(experiment="experiment027",graphs=args.graphs,method=args.method,
              phase=args.phase,seed=args.seed,workers=args.workers,
              rounds=args.rounds,candidates=args.candidates,
              new_optimizer_only=True,solve_eva=False,
              historical_baseline_root=str(args.baseline_root))
    if args.dry_run:
        print(json.dumps(plan,indent=2));return
    if args.workers==1 or len(args.graphs)==1:
        rows=[s for g in args.graphs for s in run_graph(g,args)]
    else:
        rows=[]
        with ProcessPoolExecutor(max_workers=min(args.workers,len(args.graphs))) as pool:
            future={pool.submit(run_graph,g,args):g for g in args.graphs}
            for f in as_completed(future):
                graph=future[f]
                results=f.result()
                print(json.dumps(dict(graph=graph,results=results)),flush=True)
                rows.extend(results)
    print(json.dumps(dict(plan=plan,results=rows),indent=2))


if __name__=="__main__":
    main()
