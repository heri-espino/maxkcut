"""Experiment023: missing G-set rounding families, controlled common refinement.

Run with a complete maxkcut repository checkout. This program ONLY reads
preexisting original G-set input and verified nonconvex EVA factors; it never
solves an SDP or modifies Experiment002–022 archives. Every completed candidate
is independently rescored, fingerprinted, and stored atomically. Method-level
wall time, start evaluations, timeout and failure statuses are explicit.

The 'same refinement' comparison does NOT automatically imply equal compute:
MST preprocessing is deterministic and potentially O(n² p); EM, CEM and
hyperplane search have distinct internal candidate-evaluation budgets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .eva_rounding_cli import _factor_and_case, _round_one
from .io import edge_arrays
from .objective import cut_weight
from .local_search import improve_by_flips
from .eva_elite_search import optimize_kl_fast, optimize_elite

ROOT = Path("results/gset/experiment023-method-topology")
FACTOR_DIRS = (
    Path("results/gset/experiment021-complete-71-gset/eva-numba"),
    Path("results/gset/experiment003-eva-numba"),
    Path("results/gset/experiment002-eva"),
)
METHODS = (
    "control-paired", "gw-threshold", "cutaware-kmeans", "hyperplane-sweep",
    "multigroup-k4", "multigroup-k8", "em-hard", "em-cem", "em-gibbs",
    "mst-l2-longest", "mst-l2-balanced", "mst-l2-diameter",
    "mst-l2-bestcut", "mst-l2-multicut", "mst-l1-bestcut", "mst-linf-bestcut",
    "control-tabu",  # distinct *refiner*, separately flagged
)
MST = tuple(x for x in METHODS if x.startswith("mst-"))
GRAPH_IDS = tuple([f"G{i}" for i in range(1, 68)] + ["G70", "G72", "G77", "G81"])
METHOD_BUDGETS = {"pilot": dict(restarts=2, seconds=120., line_directions=8,
                                cem_generations=4, gibbs_sweeps=10, em_stages=2,
                                em_iters=4, multigroup_iters=15, kl_passes=5,
                                ils_cycles=5, ils_kl_passes=3),
                  "full": dict(restarts=100, seconds=240., line_directions=32,
                               cem_generations=12, gibbs_sweeps=50, em_stages=4,
                               em_iters=8, multigroup_iters=60, kl_passes=30,
                               ils_cycles=80, ils_kl_passes=8)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    def convert(value):
        if isinstance(value, np.integer): return int(value)
        if isinstance(value, np.floating): return float(value)
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, Path): return str(value)
        raise TypeError(f"Cannot archive {type(value).__name__}")
    temp.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False, default=convert) + "\n", encoding="utf-8")
    os.replace(temp, path)


def select_factor(graph, dirs=FACTOR_DIRS):
    for directory in dirs:
        if (directory / "cases" / f"{graph}.json").is_file() and (directory / "factors" / f"{graph}.npz").is_file():
            return directory
    raise FileNotFoundError(f"No complete archived EVA source factor for {graph}")


def config(phase):
    if phase not in METHOD_BUDGETS:
        raise ValueError(f"Unknown execution phase {phase}")
    return METHOD_BUDGETS[phase].copy()


def fingerprint(graphsha, factorsha, method, phase, cfg):
    return hashlib.sha256(json.dumps(dict(graph_sha=graphsha, factor_sha=factorsha,
                     method=method, phase=phase, config=cfg, code_version=1),
                     sort_keys=True).encode()).hexdigest()[:20]


def _score(edges, labels):
    x = np.asarray(labels, dtype=np.int8)
    if x.ndim != 1 or not np.isin(x, [0, 1]).all():
        raise ValueError("Nonbinary partition")
    return float(cut_weight(edges, x))


def tree_for(graph, y, edges, metric, folder, graph_sha, factor_sha):
    """Cache an exact MST, with original factor-identity and tree SHA checks."""
    from .eva_mst import mst_binary_partition
    p = folder / "trees" / f"{metric}.npz"
    meta = p.with_suffix(".json")
    if p.is_file() and meta.is_file():
        settings = json.loads(meta.read_text(encoding="utf8"))
        if (settings.get("factor_sha") != factor_sha or settings.get("graph_sha") != graph_sha
                or settings.get("sha256") != sha(p)):
            raise ValueError(f"MST cache identity mismatch {p}")
        with np.load(p, allow_pickle=False) as loaded:
            parent, lengths = loaded["parent"].copy(), loaded["lengths"].copy()
        if len(parent) != len(y) or len(lengths) != len(y) or int(parent[0]) != -1:
            raise ValueError("MST cache invalid shape/parent")
        return parent, lengths, True, settings.get("build_seconds", 0.)
    _, diag, parent, lengths = mst_binary_partition(y, metric)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.stem + ".temporary.npz")
    np.savez_compressed(temp, parent=parent, lengths=lengths)
    os.replace(temp, p)
    atomic_json(meta, dict(factor_sha=factor_sha, graph_sha=graph_sha,
                          sha256=sha(p), metric=metric,
                          build_seconds=diag["seconds_excluding_jit_compilation"]))
    return parent, lengths, False, diag["seconds_excluding_jit_compilation"]


def initial(graph, y, edges, method, restart, cfg, folder, graphsha, factorsha):
    """One method-specific hard partition, never an unverified elite copy."""
    number = int(folder.name[1:])
    rng = np.random.default_rng(np.random.SeedSequence([2026, number, restart, 2301,
                                                         METHODS.index(method)]))
    diag = dict(method=method, initial_cut_evaluations="not-instrumented")
    if method in ("control-paired", "control-tabu"):
        labels, d = _round_one(y, graph, edges, "K-MeansNM_EVA", number, restart, 2026)
        diag["rounder"] = "K-MeansNM_EVA"
    elif method == "gw-threshold":
        from .eva_cutaware_kmeans import optimal_threshold
        labels, d = optimal_threshold(graph, y @ rng.standard_normal(y.shape[1]))
    elif method == "cutaware-kmeans":
        from .eva_cutaware_kmeans import improve_kmeans_direction
        labels, d = improve_kmeans_direction(y, graph, rng, direction_trials=12,
                            kmeans_iters=cfg["multigroup_iters"], perturb_scales=(.03,.08,.16,.35))
    elif method == "hyperplane-sweep":
        from .eva_direct_rounding import optimize_hyperplane
        _, labels, d = optimize_hyperplane(graph, y, rng, directions=cfg["line_directions"])
    elif method.startswith("multigroup-"):
        from .eva_multigroup_kmeans import optimize_multigroup_kmeans
        k = int(method.rsplit("k", 1)[1])
        candidates, d = optimize_multigroup_kmeans(y, graph, rng, k=k, init="paired",
                                      max_iter=cfg["multigroup_iters"], top_merges=1)
        labels = candidates[0]["labels"]
    elif method.startswith("em-"):
        from .eva_soft_em_bernoulli import soft_em, cem, gibbs_anneal
        probs, labels, d = soft_em(graph, y, rng, mode="mean", stages=cfg["em_stages"],
                                   em_iters=cfg["em_iters"])
        if method == "em-cem":
            _, labels, d2 = cem(graph, probs, rng, generations=cfg["cem_generations"],
                                  population=48)
            diag["cem_seconds"] = d2["seconds"]
        elif method == "em-gibbs":
            labels, d2 = gibbs_anneal(graph, labels, rng, sweeps=cfg["gibbs_sweeps"])
            diag["gibbs_seconds"] = d2["seconds"]
    elif method in MST:
        from .eva_mst_center import balanced_edge, diameter_center_edge, best_cut_edge, labels_cutting_edge
        from .eva_mst_multicut import multi_edge_candidates
        from .eva_mst import _cut_longest_tree_edge
        metric = method.split("-")[1]
        parent, lengths, cache_reused, cost = tree_for(graph,y,edges,metric,folder,graphsha,factorsha)
        diag.update(mst_preprocess_reused=cache_reused, mst_build_seconds=cost,
                    mst_cost_amortization="shared family factor")
        if method.endswith("-longest"):
            labels, _, _ = _cut_longest_tree_edge(parent, lengths)
        elif method.endswith("-balanced"):
            cut, _ = balanced_edge(parent, lengths)
            labels = labels_cutting_edge(parent, cut)
        elif method.endswith("-diameter"):
            cut, _ = diameter_center_edge(parent, lengths)
            labels = labels_cutting_edge(parent, cut)
        elif method.endswith("-bestcut"):
            cut, _ = best_cut_edge(parent, lengths, edges)
            labels = labels_cutting_edge(parent, cut)
        elif method.endswith("-multicut"):
            run = list(multi_edge_candidates(graph,parent,lengths,[1,4,16],
                       coarse_restarts=4,coarse_sweeps=10,flip_sweeps=10,
                       seed=2026+restart,graph_number=number,metric_index=2))
            labels = max(run, key=lambda x:x["best_coarse_cut"])["best_coarse_labels"]
        else:
            raise ValueError(method)
    else:
        raise ValueError(f"Unknown method: {method}")
    return np.asarray(labels,dtype=np.int8), diag


def refinement(graph, edges, labels, method, restart, graphnumber, cfg):
    """Common seeded 20-Flip -> 30-KL -> 80-ILS for all new rounders.

    Tabu is an explicitly separate postprocessor comparator, not an
    initializer or an equal-refinement comparison.
    """
    x = np.asarray(labels,dtype=np.int8)
    stages = {"Round": _score(edges,x)}
    rng = lambda tag: np.random.default_rng(np.random.SeedSequence([2026,graphnumber,restart,tag]))
    f, _ = improve_by_flips(graph, x.copy(), rng(1313), max_sweeps=20)
    stages["Flip"] = _score(edges,f)
    k, _ = optimize_kl_fast(graph, f.copy(), rng(101), passes=cfg["kl_passes"])
    if _score(edges,k) < stages["Flip"]-1e-8: k = f.copy()
    stages["KL"] = _score(edges,k)
    if method == "control-tabu":
        fin, d = optimize_elite(graph,k,[(k,stages["KL"])],rng(1401),method="Tabu-KL",
                                cycles=cfg["ils_cycles"],kl_passes=cfg["ils_kl_passes"],
                                kick_sizes=(8,16,32,64),temperature=12,
                                tabu_steps=1000,tabu_stagnation=250)
    else:
        fin, d = optimize_elite(graph,k,[(k,stages["KL"])],rng(1401),method="ILS-KL",
                                cycles=cfg["ils_cycles"],kl_passes=cfg["ils_kl_passes"],
                                kick_sizes=tuple(min(graph.shape[0],v) for v in (8,16,32,64)),
                                temperature=12)
    stages["Final"] = _score(edges,fin)
    for a,b in zip(("Round","Flip","KL"),("Flip","KL","Final")):
        if stages[b]+1e-8<stages[a]:raise AssertionError(f"Nonmonotone {a}->{b}")
    return fin,stages,{"ils_cycles_completed":d.get("cycles_completed"),
                       "trace":d.get("trace", []),
                       "best_events":d.get("best_events", []),
                       "refiner": "Tabu-KL" if method=="control-tabu" else "ILS-KL"}, {"Round":x, "Flip":f, "KL":k, "Final":fin}


def write_case(folder, method, restart, labels, stages, info, partitions=None):
    """Atomic individual archive; checksum binds labels to recorded cut."""
    path = folder / method / f"{restart:04d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_name(path.stem + ".temporary.npz")
    part=partitions or {"Final":labels}
    np.savez_compressed(target, **{stage:np.asarray(lab,dtype=np.uint8) for stage,lab in part.items()})
    os.replace(target,path)
    doc = dict(info, stages=stages, labels_sha256=sha(path))
    atomic_json(path.with_suffix(".json"),doc)
    return doc


def verify_case(folder, method, restart, graph, edges, ident):
    path = folder / method / f"{restart:04d}.npz"
    meta = path.with_suffix(".json")
    if not path.is_file() or not meta.is_file():return None
    data = json.loads(meta.read_text(encoding="utf8"))
    if data.get("fingerprint") != ident or data.get("labels_sha256") != sha(path):
        raise ValueError(f"Incompatible or corrupted case {path}")
    with np.load(path,allow_pickle=False) as src:
        if set(src.files)!={"Round","Flip","KL","Final"}:
            raise ValueError(f"Incomplete four-stage archive {path}")
        for stage in ("Round","Flip","KL","Final"):
            labels=src[stage]
            if (labels.shape != (graph.shape[0],) or not np.isin(labels,[0,1]).all()
                    or abs(_score(edges,labels)-data["stages"][stage])>1e-8):
                raise ValueError(f"Invalid stored {stage} signed cut {path}")
    return data


def run_case(path, method, phase, *, output=ROOT, budget_override=None):
    """Check existing immutable cases and append only new verified runs."""
    if method not in METHODS:raise ValueError(method)
    cfg=config(phase)
    if budget_override is not None:cfg["seconds"]=float(budget_override)
    started=time.monotonic()
    factor=select_factor(path.stem)
    graph,y,_,graph_sha,factor_sha=_factor_and_case(path,factor)
    edges=edge_arrays(graph)
    folder=Path(output)/phase/path.stem
    ident=fingerprint(graph_sha,factor_sha,method,phase,cfg)
    summary=folder/method/"status.json"
    previous=json.loads(summary.read_text(encoding="utf8")) if summary.is_file() else None
    if previous and previous.get("fingerprint")!=ident:
        raise ValueError("Method checkpoint belongs to another experiment fingerprint")
    completed=0;best=-float("inf");attempted=0
    elapsed_before=float(previous.get("cumulative_wall_seconds",0.)) if previous else 0.
    # Budget is a cumulative allowance across resumptions, not reset each time.
    if previous and previous.get("status") in ("budget_limited", "completed_restarts"):
        if elapsed_before >= cfg["seconds"] or previous.get("status")=="completed_restarts":
            return previous
    for restart in range(cfg["restarts"]):
        if elapsed_before+time.monotonic()-started>=cfg["seconds"]:
            break
        cached=verify_case(folder,method,restart,graph,edges,ident)
        if cached:
            completed+=1;best=max(best,cached["stages"]["Final"]);continue
        t0=time.monotonic()
        seed,diag=initial(graph,y,edges,method,restart,cfg,folder,graph_sha,factor_sha)
        raw=_score(edges,seed)
        final,stages,refine,partitions=refinement(graph,edges,seed,method,restart,int(path.stem[1:]),cfg)
        d=dict(fingerprint=ident,graph=path.stem,method=method,phase=phase,
               restart=restart,graph_sha256=graph_sha,factor_sha256=factor_sha,
               factor_source=str(factor),raw_score=raw,method_diagnostics=diag,
               refinements=refine,seconds=time.monotonic()-t0)
        # Explicitly audit binary labels from the finalized partition
        stored=write_case(folder,method,restart,final,stages,d,partitions)
        verified=verify_case(folder,method,restart,graph,edges,ident)
        if verified is None:raise AssertionError("Failed to reload saved result")
        completed+=1;attempted+=1;best=max(best,stages["Final"])
        atomic_json(summary,dict(status="in_progress",graph=path.stem,method=method,
              phase=phase,fingerprint=ident,graph_sha256=graph_sha,factor_sha256=factor_sha,
              completed=completed,newly_completed=attempted,best=best,wall_seconds=time.monotonic()-started,
              cumulative_wall_seconds=elapsed_before+time.monotonic()-started,
              note="Not an equal-cost claim: preprocessing and internal proposals differ"))
    elapsed=time.monotonic()-started
    status=("no_candidate" if completed==0 else
            "completed_restarts" if completed==cfg["restarts"] else "budget_limited")
    out=dict(status=status,graph=path.stem,method=method,phase=phase,fingerprint=ident,
             graph_sha256=graph_sha,factor_sha256=factor_sha,completed=completed,
             newly_completed=attempted,requested=cfg["restarts"],best=best if completed else None,
             wall_seconds=elapsed,cumulative_wall_seconds=elapsed_before+elapsed,
             budget_seconds=cfg["seconds"],over_budget=elapsed_before+elapsed>cfg["seconds"],
             same_refiner=method!="control-tabu",cost_comparison="descriptive_only_until_equal_budget_analysis")
    atomic_json(summary,out)
    return out


def orchestrate(graph, phase, methods, budget_seconds, method_timeout_seconds, *, output=ROOT):
    """Use fresh subprocess per method, kill it safely on hard timeout."""
    path=Path("data")/(graph+".txt")
    if not path.is_file():raise FileNotFoundError(path)
    status=[]
    def one(method):
        command=[sys.executable,"-m","maxkcut.experiment023_methods","single","--graph",graph,
                 "--phase",phase,"--method",method,"--output",str(output),
                 "--budget-seconds",str(budget_seconds)]
        began=time.monotonic()
        try:
            subprocess.run(command,check=True,timeout=method_timeout_seconds,
                           env={**os.environ,"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1",
                                "MKL_NUM_THREADS":"1","NUMBA_NUM_THREADS":"1"})
            row=json.loads((Path(output)/phase/graph/method/"status.json").read_text(encoding="utf8"))
        except subprocess.TimeoutExpired:
            row=dict(status="hard_timeout",method=method,graph=graph,phase=phase,
                     timeout_seconds=method_timeout_seconds,completed_saved_cases="check method dir")
        except subprocess.CalledProcessError as err:
            row=dict(status="failed",method=method,graph=graph,phase=phase,
                     error=f"exit code {err.returncode}",completed_saved_cases="check method dir")
        row["orchestrator_seconds"]=time.monotonic()-began
        atomic_json(Path(output)/phase/graph/"statuses"/(method+".json"),row)
        print(json.dumps({"graph":graph,"method":method,"status":row["status"]}),flush=True)
        return row
    # Run methods whose initializers do not share MST cache concurrently.
    fast=[x for x in methods if x not in MST]
    trees=[x for x in methods if x in MST]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(one,m):m for m in fast}
        for fut in as_completed(futures):status.append(fut.result())
    # MST metrics share an independently validated tree and are serialized to
    # avoid mutating the same cache concurrently on the Windows runner.
    for m in trees:status.append(one(m))
    return status


def aggregate(output=ROOT, phase="full"):
    base=Path(output)/phase
    rows=[]
    for graph in GRAPH_IDS:
        for method in METHODS:
            p=base/graph/"statuses"/(method+".json")
            if p.is_file():
                src=json.loads(p.read_text(encoding="utf8"))
                rows.append(dict(graph=graph,method=method,status=src["status"],
                                 completed=src.get("completed"),best=src.get("best"),
                                 wall_seconds=src.get("wall_seconds"),
                                 over_budget=src.get("over_budget")))
            else:rows.append(dict(graph=graph,method=method,status="not_run",completed=None,
                                  best=None,wall_seconds=None,over_budget=None))
    report=dict(phase=phase,graphs=71,methods=len(METHODS),
                complete_graph_method_rows=sum(x["status"] in ("completed_restarts","budget_limited")
                   and (x["completed"] or 0)>0 and not x["over_budget"] for x in rows),
                status_counts={s:sum(x["status"]==s for x in rows) for s in sorted({x["status"] for x in rows})},
                note="No superiority claim without verifying statuses, exact original cut partitions, and equal-budget accounting")
    folder=Path(output)/"analysis"
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/f"{phase}_coverage.csv").open("w",newline="",encoding="utf8") as stream:
        wr=csv.DictWriter(stream,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    atomic_json(folder/f"{phase}_coverage.json",report)
    return report



def summarize_topology(output=ROOT, phase="full"):
    """Strict topology outcome table: no win counts on incomplete graph rows.

    Graphs are the statistical units. Only comparing methods with the
    same ILS refinement; the Tabu postprocessor is reported separately.
    Historical Experiment021 is NOT merged into this dataset.
    """
    from .graph_structure_methods import labels as graph_labels
    base=Path(output)/phase
    records=[]
    full=[]
    for graph in GRAPH_IDS:
        family,weight_class=graph_labels(graph)
        graph_methods=[]
        for method in METHODS:
            f=base/graph/"statuses"/(method+".json")
            x=json.loads(f.read_text(encoding="utf8")) if f.is_file() else {"status":"not_run"}
            valid=(x.get("status") in ("completed_restarts","budget_limited")
                   and x.get("completed",0)>0 and not x.get("over_budget",True)
                   and x.get("best") is not None)
            graph_methods.append(dict(graph=graph,family=family,weight_type=weight_class,
                          method=method,status=x.get("status","not_run"),eligible=bool(valid),
                          completed=x.get("completed",0),observed_best=x.get("best"),
                          time_seconds=x.get("cumulative_wall_seconds",x.get("wall_seconds")),
                          is_tabu=(method=="control-tabu")))
        candidates=[x for x in graph_methods if not x["is_tabu"]]
        full_rows=all(x["eligible"] for x in candidates)
        base_row=next(x for x in candidates if x["method"]=="control-paired")
        base_score=base_row["observed_best"] if base_row["eligible"] else None
        win=max(x["observed_best"] for x in candidates) if full_rows else None
        ties=sum(x["observed_best"]==win for x in candidates) if full_rows else 0
        for x in graph_methods:
            x["full_graph_comparison_eligible"]=bool(full_rows)
            x["shared_win"]=(int(x["observed_best"]==win) if full_rows and not x["is_tabu"] else None)
            x["fractional_win"]=(1/ties if full_rows and not x["is_tabu"]
                                  and x["observed_best"]==win else 0. if full_rows and not x["is_tabu"] else None)
            x["gain_vs_new_control"]=(x["observed_best"]-base_score if x["eligible"]
                                     and base_score is not None else None)
            records.append(x)
        if full_rows:full.append(graph)
    dest=Path(output)/"analysis"
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/f"{phase}_topology_methods.csv").open("w",newline="",encoding="utf8") as file:
        cols=list(records[0]);writer=csv.DictWriter(file,fieldnames=cols)
        writer.writeheader();writer.writerows(records)
    groups=[]
    for family in ("random","planar","toroidal"):
        for method in METHODS:
            x=[r for r in records if r["family"]==family and r["method"]==method]
            valid=[r for r in x if r["eligible"]]
            comparable=[r for r in x if r["full_graph_comparison_eligible"] and not r["is_tabu"]]
            groups.append(dict(family=family,method=method,graphs_total=len(x),
                               successful_within_budget=len(valid),
                               complete_comparison_graphs=len(comparable),
                               shared_wins=sum(r["shared_win"] or 0 for r in comparable),
                               fractional_wins=sum(r["fractional_win"] or 0 for r in comparable),
                               mean_gain_vs_control=(sum(r["gain_vs_new_control"] for r in valid)/len(valid)
                                                     if valid and all(r["gain_vs_new_control"] is not None for r in valid) else None)))
    with (dest/f"{phase}_topology_summary.csv").open("w",newline="",encoding="utf8") as file:
        cols=list(groups[0]);writer=csv.DictWriter(file,fieldnames=cols)
        writer.writeheader();writer.writerows(groups)
    out=dict(phase=phase,graph_count=len(GRAPH_IDS),total_graph_method_cells=len(records),
             graphs_with_complete_budget_eligible_ils_comparison=len(full),
             graph_ids_with_complete_ils_comparison=full,
             topology_method_summary_file=f"{phase}_topology_summary.csv",
             caveat="Generator families are not independent; no equal-evaluation-count guarantee; Tabu uses different refiner; require hardware and budget audit")
    atomic_json(dest/f"{phase}_topology_report.json",out)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("single","batch","aggregate","inventory","analyze"))
    parser.add_argument("--graph",choices=GRAPH_IDS,default="G11")
    parser.add_argument("--phase",choices=("pilot","full"),default="pilot")
    parser.add_argument("--method",choices=METHODS,default="control-paired")
    parser.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    parser.add_argument("--output",type=Path,default=ROOT)
    parser.add_argument("--budget-seconds",type=float)
    parser.add_argument("--timeout-seconds",type=float)
    args=parser.parse_args()
    if args.action=="inventory":
        rows=[]
        for graph in GRAPH_IDS:
            rows.append(dict(graph=graph,source=str(select_factor(graph))))
        print(json.dumps({"graph_count":len(rows),"factors":rows},indent=2))
        return
    if args.action=="aggregate":print(json.dumps(aggregate(args.output,args.phase),indent=2));return
    if args.action=="analyze":print(json.dumps(summarize_topology(args.output,args.phase),indent=2));return
    sec=args.budget_seconds or config(args.phase)["seconds"]
    if args.action=="single":
        result=run_case(Path("data")/(args.graph+".txt"),args.method,args.phase,
                       output=args.output,budget_override=sec)
    else:
        timeout=args.timeout_seconds or sec*1.75+90
        result=orchestrate(args.graph,args.phase,args.methods,sec,timeout,output=args.output)
    print(json.dumps(result,indent=2,default=str))

if __name__=="__main__":main()
