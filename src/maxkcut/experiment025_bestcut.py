"""Experiment025: maximize final signed Max-Cut over 16 common GW starts.

This is a quality-first follow-up to Experiment024. Each graph shares the exact
same 16 saved GW Gaussian unit normals across ALL eighteen method protocols.
Every one of the 16 method-specific raw candidates is independently refined:
    raw -> 20 positive-flip sweeps -> 30 KL passes -> 80 ILS-KL cycles.
The winner is selected by BEST VERIFIED CUT FOUND AT ANY ITERATION,
not the last iterate or best raw cut.
There is NO METHOD-LEVEL wall-time budget, timeout, or early-quality cutoff.

Input EVA factors and historical outputs are immutable. All seed-level
partitions and diagnostics checkpoint after each successful independent start,
so Windows runner interruptions can be resumed without rerunning completed
seeds. The unavoidable GitHub Actions job timeout is a platform safeguard,
not a method ranking metric. Cross-method algorithmic work still differs.
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
from pathlib import Path

import numpy as np

from .experiment024_paired_gw16 import (
    GRAPH_IDS, METHODS, MST, SEED, STARTS, directions_archive,
    generate_candidate,
)
from .experiment023_methods import (
    select_factor, write_case, verify_case,
    atomic_json, sha, _score,
)
from .eva_rounding_cli import _factor_and_case
from .io import edge_arrays
from .experiment023_references import load_references, compare_cut
from .graph_structure_methods import labels as graph_family
from .experiment025_journal import IterationJournal, refine_with_journal

ROOT=Path("results/gset/experiment025-gw16-bestcut-quality")

# Method-specific algorithmic search settings, NOT time limits.
# Full common refinement matches the historical full KL+ILS pipeline.
SETTINGS={
    "pilot":dict(starts=16,kmeans_iters=8,direction_trials=2,
        line_directions=2,em_stages=1,em_iters=2,
        cem_generations=2,cem_population=16,gibbs_sweeps=2,
        kl_passes=3,ils_cycles=2,ils_kl_passes=2),
    "full":dict(starts=16,kmeans_iters=100,direction_trials=96,
        line_directions=250,em_stages=8,em_iters=20,
        cem_generations=30,cem_population=96,gibbs_sweeps=150,
        kl_passes=30,ils_cycles=80,ils_kl_passes=8),
}
PROTOCOL="experiment025-gw16-all16-anytime-quality-v2"


def fingerprint(graph_sha,factor_sha,direction_sha,method,phase,settings):
    raw=json.dumps(dict(protocol=PROTOCOL,graph_sha=graph_sha,
         factor_sha=factor_sha,direction_sha=direction_sha,
         method=method,phase=phase,settings=settings),
         sort_keys=True)
    return hashlib.sha256(raw.encode("utf8")).hexdigest()[:24]


def _check_status(path,identity):
    if not path.is_file(): return None
    obj=json.loads(path.read_text(encoding="utf8"))
    if obj.get("fingerprint")!=identity:
        raise ValueError(f"Incompatible saved result for current protocol: {path}")
    return obj


def result_for_seed(folder, method, index, graph, edges, identity):
    """Validate every stored stage's binary partition and signed cut."""
    return verify_case(folder,method,index,graph,edges,identity)


def _status(graph,method,phase,identity,completed,details,started):
    ends=[float(x["best_seen_cut"]) for x in details]
    raws=[float(x["stages"]["Round"]) for x in details]
    best=int(np.argmax(ends)) if len(details)==16 else None
    return dict(graph=graph,method=method,phase=phase,fingerprint=identity,
       status="complete_16_best_seen" if len(details)==16 else "partially_refined",
       starts_completed=len(details),starts_requested=16,
       all_starts_independently_refined=len(details)==16,
       selected_seed_index=best,
       best_final=max(ends) if ends else None,
       best_seen=max(ends) if ends else None,
       best_initial=max(raws) if raws else None,
       best_last_ils_candidate=max([x.get("last_iteration_cut",x["stages"]["Final"]) for x in details]) if details else None,
       mean_final=float(np.mean(ends)) if ends else None,
       mean_best_seen=float(np.mean(ends)) if ends else None,
       mean_initial=float(np.mean(raws)) if raws else None,
       wall_seconds_this_invocation=time.monotonic()-started,
       selection_metric="max_best_seen_cut_over_all_iterations_and_16_refinements",
       method_level_time_cutoff=None,
       deterministic_mst_hybrid=method in MST)


def run_method(graph,method,phase,*,root=ROOT,max_new_seeds=None):
    if graph not in GRAPH_IDS or method not in METHODS or phase not in SETTINGS:
        raise ValueError("Invalid graph/method/phase")
    cfg=SETTINGS[phase].copy()
    if cfg["starts"]!=STARTS or STARTS!=16:raise AssertionError("Expected 16 fixed starts")
    started=time.monotonic()
    source=select_factor(graph)
    g,y,_,graph_sha,factor_sha=_factor_and_case(Path("data")/(graph+".txt"),source)
    edges=edge_arrays(g)
    folder=Path(root)/phase/graph
    normals,gw_labels,meta=directions_archive(folder,y,graph,graph_sha,factor_sha)
    if len(normals)!=16 or len(gw_labels)!=16:raise AssertionError("Missing shared GW normals")
    identity=fingerprint(graph_sha,factor_sha,meta["direction_rows_sha256"],
                         method,phase,cfg)
    target=folder/method
    target.mkdir(parents=True,exist_ok=True)
    status_file=target/"status.json"
    _check_status(status_file,identity)
    details=[]
    newly_completed=0
    if max_new_seeds is not None and max_new_seeds<1:
        raise ValueError('max_new_seeds must be positive')
    for index,(normal,gw) in enumerate(zip(normals,gw_labels)):
        if max_new_seeds is not None and newly_completed>=max_new_seeds:
            break
        saved=result_for_seed(folder,method,index,g,edges,identity)
        if saved is None:
            start=time.monotonic()
            journal=IterationJournal(target/"progress"/f"{index:04d}",edges,
                                     g.shape[0],identity)
            raw,raw_score,extra=generate_candidate(
                g,y,edges,method,normal,gw,index,cfg,folder,graph_sha,factor_sha,
                on_iteration=journal.observe)
            # Some iterative methods (notably soft EM) can have an intermediate
            # hard cut better than their final membership classification.
            # Use that *best seen* original signed cut as the refinement seed.
            initial_best, initial_cut=journal.assert_best()
            if initial_cut>raw_score+1e-8:
                raw,raw_score=initial_best,initial_cut
                extra["better_intermediate_initializer_used"]=True
            else:
                extra["better_intermediate_initializer_used"]=False
            final,stages,ref_info,partitions=refine_with_journal(
                g,edges,raw,method,index,int(graph[1:]),cfg,journal)
            if not np.isclose(stages["Round"],raw_score,atol=1e-8,rtol=0):
                raise AssertionError("Selected best initializer/raw cut mismatch")
            metadata=dict(graph=graph,method=method,phase=phase,
                index=index,fingerprint=identity,
                graph_sha256=graph_sha,factor_sha256=factor_sha,
                shared_direction_sha256=meta["direction_rows_sha256"],
                common_gw_start_cut=_score(edges,gw),
                raw_cut=raw_score,final_cut=stages["Final"],
                best_seen_cut=journal.best_cut,
                last_iteration_cut=(ref_info["ils_trace"][-1]["candidate_cut"]
                                    if ref_info["ils_trace"] else stages["KL"]),
                best_seen_event=journal.best_event,
                progress_events=journal.events,
                progress_improvements=journal.improvements,
                progress_dir=str(journal.path),
                method_details=extra,refinement_details=ref_info,
                seed_seconds=time.monotonic()-start,
                scheme="GW16; all iterations audited; best seen across 16 starts")
            write_case(folder,method,index,final,stages,metadata,partitions)
            saved=result_for_seed(folder,method,index,g,edges,identity)
            if saved is None:raise AssertionError("Seed checkpoint not persisted")
            newly_completed+=1
        # Recheck saved seed stage scores against the immutable graph and factor.
        if saved.get("graph_sha256")!=graph_sha or saved.get("factor_sha256")!=factor_sha:
            raise ValueError("Saved seed belongs to another graph or factor")
        if saved.get("index")!=index:
            raise ValueError("Saved seed index mismatch")
        if float(saved["best_seen_cut"])+1e-8 < max(saved["stages"].values()):
            raise AssertionError("Saved best-seen cut lower than a stage checkpoint")
        details.append(saved)
        # Every success is resumable on the self-hosted runner; no fixed time limit.
        partial=_status(graph,method,phase,identity,len(details),details,started)
        atomic_json(status_file,partial)
        atomic_json(folder/"statuses"/(method+".json"),partial)
    status=_status(graph,method,phase,identity,len(details),details,started)
    if len(details)<16:
        return status
    refs,_=load_references()
    best=float(status["best_final"])
    status.update(reference_cut=refs[graph]["reference_cut"],
        reference_comparison=compare_cut(best,refs[graph]),
        graph_sha256=graph_sha,factor_sha256=factor_sha,
        shared_directions_sha256=meta["direction_rows_sha256"],
        evaluated_independent_refinements=16,
        original_signed_cut_rescored=True,
        best_final_partition_path=f"{int(status['selected_seed_index']):04d}.npz",
        winning_progress_log=f"{method}/progress/{int(status['selected_seed_index']):04d}/events.jsonl",
        winning_best_meta=f"{method}/progress/{int(status['selected_seed_index']):04d}/best.json")
    atomic_json(status_file,status)
    atomic_json(folder/"statuses"/(method+".json"),status)
    return status


def batch(graph,phase,*,root=ROOT):
    """No method timeout. Every method's failure is saved, and next runs retry."""
    rows=[]
    for method in METHODS:
        command=[sys.executable,"-m","maxkcut.experiment025_bestcut",
          "single","--graph",graph,"--method",method,"--phase",phase,"--root",str(root)]
        logs=Path(root)/phase/graph/"logs"
        logs.mkdir(parents=True,exist_ok=True)
        started=time.monotonic()
        try:
            cp=subprocess.run(command,check=True,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,errors="replace",
                env={**os.environ,"OPENBLAS_NUM_THREADS":"1",
                     "OMP_NUM_THREADS":"1","NUMBA_NUM_THREADS":"1",
                     "MKL_NUM_THREADS":"1"})
            transcript=cp.stdout or ""
            r=json.loads((Path(root)/phase/graph/method/"status.json").read_text(encoding="utf8"))
        except subprocess.CalledProcessError as e:
            transcript=e.stdout or ""
            r=dict(graph=graph,method=method,phase=phase,status="failed",
                   exit_code=e.returncode,diagnostic="see method log and seed checkpoints")
        (logs/(method+".log")).write_text(transcript[-25000:],encoding="utf8")
        r["orchestrator_seconds"]=time.monotonic()-started
        atomic_json(Path(root)/phase/graph/"statuses"/(method+".json"),r)
        rows.append(r)
        print(json.dumps({k:r.get(k) for k in (
            "graph","method","status","starts_completed","best_final")}),flush=True)
    return rows


def summarize(*,root=ROOT,phase="full"):
    refs,_=load_references()
    folder=Path(root)/"analysis"
    folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for graph in GRAPH_IDS:
        fam,sign=graph_family(graph)
        root_g=Path(root)/phase/graph
        by_method={}
        for method in METHODS:
            p=root_g/"statuses"/(method+".json")
            r=json.loads(p.read_text(encoding="utf8")) if p.exists() else {}
            by_method[method]=r
        baseline=by_method["gw-rr"]
        base=(float(baseline["best_seen"]) if baseline.get("status")=="complete_16_best_seen"
            and baseline.get("starts_completed")==16 else None)
        eligible=[float(r["best_seen"]) for r in by_method.values()
                  if r.get("status")=="complete_16_best_seen" and r.get("starts_completed")==16]
        peak=max(eligible) if eligible else None
        for method in METHODS:
            r=by_method[method]
            valid=r.get("status")=="complete_16_best_seen" and r.get("starts_completed")==16
            val=float(r["best_seen"]) if valid else None
            rows.append(dict(graph=graph,family=fam,weight_type=sign,method=method,
                status=r.get("status","not_run"),refined_starts=r.get("starts_completed",0),
                best_seen_cut=val if valid else "",mean_best_seen=r.get("mean_best_seen","") if valid else "",
                best_last_ils_candidate=r.get("best_last_ils_candidate","") if valid else "",
                reference_cut=refs[graph]["reference_cut"],
                gap_to_reference=refs[graph]["reference_cut"]-val if valid else "",
                gw_rr_full16=base if base is not None else "",
                gain_vs_gw=val-base if valid and base is not None else "",
                tied_best=int(val==peak) if valid else "",
                family_warning="GW-anchored MST hybrid" if method in MST else "",
                wall_seconds=r.get("wall_seconds_this_invocation","")))
    with (folder/(phase+"_best_final_by_graph.csv")).open("w",newline="",encoding="utf8") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    complete=sum(r["status"]=="complete_16_best_seen" and r["refined_starts"]==16 for r in rows)
    report=dict(phase=phase,graphs=71,methods=len(METHODS),starts_per_method=16,
        planned_cells=len(GRAPH_IDS)*len(METHODS),completed_cells=complete,
        planned_independent_full_refinements=len(GRAPH_IDS)*len(METHODS)*16,
        all_complete=complete==len(rows),
        primary_objective="best verified cut ever visited across all initializer and refinement iterations",
        note="No method-level runtime cutoff; incomplete/failed cells do not enter rankings")
    atomic_json(folder/(phase+"_coverage.json"),report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["single","batch","summary"])
    p.add_argument("--graph",choices=GRAPH_IDS,default="G11")
    p.add_argument("--method",choices=METHODS,default="gw-rr")
    p.add_argument("--phase",choices=tuple(SETTINGS),default="pilot")
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--max-new-seeds",type=int,default=None)
    a=p.parse_args()
    if a.action=="single":r=run_method(a.graph,a.method,a.phase,root=a.root,
                                       max_new_seeds=a.max_new_seeds)
    elif a.action=="batch":r=batch(a.graph,a.phase,root=a.root)
    else:r=summarize(root=a.root,phase=a.phase)
    print(json.dumps(r,indent=2,default=str))

if __name__=="__main__": main()
