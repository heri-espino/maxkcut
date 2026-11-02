"""Experiment024: 16 common GW hyperplanes per graph, paired across all methods.

The 16 raw candidates for every method share identical unit normals r_s and
their GW partitions. Each method receives the same (r_s, x_GW,s), but may
perform different internal work; this is paired-start fairness, NOT equal
computational cost. Refine only the best-of-16 raw candidate ONCE using the
same Flip -> KL -> ILS stages for every method except Tabu control.

MST is deterministic and has no hyperplane initialization. Its explicitly
marked GW-ANCHORED HYBRID retains the GW partition as incumbent and competes
against deterministic MST cuts. Do not mislabel it as 16 independent MST starts.

Never overwrite Experiment023 scores, factors, or partitions. Checkpoint all
16 starting directions, raw partitions, raw cuts, hashes, final stages, costs,
and failures under Experiment024. Only complete 16-of-16 method rows qualify
for the primary ranking. Original signed graph objective remains authoritative.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .io import edge_arrays
from .objective import cut_weight
from .eva_rounding_cli import _factor_and_case
from .eva_cutaware_kmeans import optimal_threshold, improve_kmeans_direction
from .eva_direct_rounding import optimize_hyperplane
from .eva_multigroup_kmeans import optimize_multigroup_kmeans
from .eva_soft_em_bernoulli import soft_em, cem, gibbs_anneal
from .eva_mst_center import balanced_edge, diameter_center_edge, best_cut_edge, labels_cutting_edge
from .eva_mst import _cut_longest_tree_edge
from .eva_mst_multicut import multi_edge_candidates
from .experiment023_methods import (
    GRAPH_IDS, METHODS, MST, select_factor, tree_for, refinement,
    atomic_json, sha, _score, verify_case, write_case,
)
from .experiment023_references import load_references, compare_cut
from .graph_structure_methods import labels as graph_family

BASE=Path("results/gset/experiment024-paired-gw16")
SEED=2026
STARTS=16
PILOT_GRAPHS=("G1","G14","G11")


def shared_directions(y, graph_number, *, seed=SEED, starts=STARTS):
    """Same full-dimensional Gaussian unit hyperplanes for every method."""
    if starts != 16:
        raise ValueError("Experiment024 pre-registration fixes exactly 16 starts")
    d=y.shape[1]
    if d<1: raise ValueError("Invalid factor dimension")
    out=[]
    for s in range(starts):
        z=np.random.default_rng(np.random.SeedSequence(
            [seed, graph_number, 2401, s])).standard_normal(d)
        norm=np.linalg.norm(z)
        if not np.isfinite(norm) or norm<1e-12:
            raise ValueError("Cannot draw a valid GW normal")
        out.append(z/norm)
    return np.ascontiguousarray(np.stack(out), dtype=np.float64)


def gw_partition(y,r):
    if r.shape!=(y.shape[1],) or not np.isfinite(r).all():
        raise ValueError("Invalid GW direction")
    return (y@r>0).astype(np.int8)


def directions_archive(folder, y, graph, graphsha, factorsha):
    """Exactly one immutable 16 x factor_rank directions file per graph."""
    folder=Path(folder)
    normals=shared_directions(y,int(graph[1:]))
    partitions=np.stack([gw_partition(y,r) for r in normals])
    file=folder/"gw16_common.npz"
    metadata=file.with_suffix(".json")
    expected=dict(graph=graph,graph_sha256=graphsha,factor_sha256=factorsha,
                  source_seed=SEED,starts=16,direction_rows_sha256=hashlib.sha256(
                  normals.tobytes()).hexdigest(),
                  gw_labels_sha256=hashlib.sha256(partitions.tobytes()).hexdigest())
    if file.exists() or metadata.exists():
        if not file.is_file() or not metadata.is_file():
            raise ValueError("Partial shared direction archive")
        original=json.loads(metadata.read_text(encoding="utf8"))
        if any(original.get(k)!=v for k,v in expected.items()) or sha(file)!=original.get("npz_sha256"):
            raise ValueError("Shared GW directions changed under same graph/factor")
        with np.load(file,allow_pickle=False) as z:
            if (not np.array_equal(z["normals"],normals) or
                    not np.array_equal(z["gw_partitions"],partitions)):
                raise ValueError("Shared GW direction contents mismatch")
    else:
        folder.mkdir(parents=True,exist_ok=True)
        temp=file.with_name("gw16_common.tmp.npz")
        np.savez_compressed(temp,normals=normals,gw_partitions=partitions)
        os.replace(temp,file)
        atomic_json(metadata,dict(expected,npz_sha256=sha(file)))
    return normals,partitions,expected


def seeded_lloyd(y,edges,r,*,max_iter=60):
    """Antipodal spherical K-means starting from +/- the exact GW normal."""
    unit=np.asarray(r,dtype=np.float64)
    centers=np.stack((-unit,unit))
    previous=None
    gw=gw_partition(y,r)
    best=gw.copy()
    best_score=_score(edges,best)
    iters=0
    for it in range(max_iter):
        labels=np.argmax(y@centers.T,axis=1).astype(np.int8)
        score=_score(edges,labels)
        if score>best_score+1e-8:
            best,best_score=labels.copy(),score
        iters=it+1
        if previous is not None and np.array_equal(labels,previous):break
        previous=labels.copy()
        for c in range(2):
            members=y[labels==c]
            if len(members):
                total=np.sum(members,axis=0)
                norm=np.linalg.norm(total)
                if norm>1e-12:centers[c]=total/norm
    return best,dict(lloyd_iterations=iters,retained_original_gw=True)


def generate_candidate(graph,y,edges,method,r,gw,s,cfg,folder,graphsha,factorsha):
    """All methods receive exactly the SAME GW direction/partition for s."""
    gn=int(folder.name[1:])
    mid=METHODS.index(method)
    rng=np.random.default_rng(np.random.SeedSequence([SEED,gn,2402,s,mid]))
    info=dict(gw_seed_index=s,shared_gw_direction=True,source="GW_RR_direction")
    if method=="gw-rr":
        candidate=gw.copy()
    elif method in ("control-paired","control-tabu"):
        candidate,d=seeded_lloyd(y,edges,r,max_iter=cfg["kmeans_iters"])
        info["lloyd_iterations"]=d["lloyd_iterations"]
    elif method=="gw-threshold":
        candidate,d=optimal_threshold(graph,y@r)
        info["conditional_sorted_prefix"]=d["sorted_prefix"]
    elif method=="cutaware-kmeans":
        candidate,d=improve_kmeans_direction(
            y,graph,rng,initial_direction=r,
            direction_trials=cfg["direction_trials"],
            kmeans_iters=cfg["kmeans_iters"],perturb_scales=(.05,.2))
        info["additional_proposals"]="cutaware trajectory and threshold searches"
    elif method=="hyperplane-sweep":
        _,candidate,d=optimize_hyperplane(
            graph,y,rng,initial=r,directions=cfg["line_directions"])
        info["line_search_directions"]=cfg["line_directions"]
    elif method.startswith("multigroup-"):
        k=int(method.rsplit("k",1)[1])
        options,d=optimize_multigroup_kmeans(
            y,graph,rng,k=k,init="paired",initial_direction=r,
            max_iter=cfg["kmeans_iters"],top_merges=1)
        candidate=np.asarray(options[0]["labels"],dtype=np.int8)
        info["extra_group_merge_evaluations"]="see method trace"
    elif method.startswith("em-"):
        # Anchor is ONLY the current GW seed, never an archived best-known
        # solution. Elite here means GW-seeded probability mass, not prior elite.
        probs,hard,d=soft_em(graph,y,rng,mode="elite",anchor=gw,
                 stages=cfg["em_stages"],em_iters=cfg["em_iters"])
        candidate=hard
        info["em_anchor"]="current shared GW partition; no historical elite"
        if method=="em-cem":
            _,candidate,d2=cem(graph,probs,rng,anchor=gw,
                  generations=cfg["cem_generations"],population=cfg["cem_population"])
            info["cem_generations"]=cfg["cem_generations"]
        elif method=="em-gibbs":
            candidate,d2=gibbs_anneal(graph,hard,rng,sweeps=cfg["gibbs_sweeps"])
            info["gibbs_sweeps"]=cfg["gibbs_sweeps"]
    elif method in MST:
        metric=method.split("-")[1]
        parent,lengths,cached,cost=tree_for(
            graph,y,edges,metric,folder,graphsha,factorsha)
        info.update(mst_is_gw_anchored_hybrid=True,independent_mst_random_draws=0,
                    original_mst_preprocessing_seconds=cost,mst_cached=cached)
        if method.endswith("-longest"):
            candidate,_,_=_cut_longest_tree_edge(parent,lengths)
        elif method.endswith("-balanced"):
            child,_=balanced_edge(parent,lengths)
            candidate=labels_cutting_edge(parent,child)
        elif method.endswith("-diameter"):
            child,_=diameter_center_edge(parent,lengths)
            candidate=labels_cutting_edge(parent,child)
        elif method.endswith("-bestcut"):
            child,_=best_cut_edge(parent,lengths,edges)
            candidate=labels_cutting_edge(parent,child)
        elif method.endswith("-multicut"):
            options=list(multi_edge_candidates(
                graph,parent,lengths,[1,4,16],coarse_restarts=2,
                coarse_sweeps=5,flip_sweeps=5,
                seed=SEED+gn*100+s,graph_number=gn,metric_index=2))
            pick=max(options,key=lambda x:x["best_coarse_cut"])
            candidate=pick["best_coarse_labels"]
        else:raise ValueError(method)
    else:
        raise ValueError(f"Unknown method {method}")
    candidate=np.asarray(candidate,dtype=np.int8)
    if candidate.shape!=gw.shape or not np.isin(candidate,[0,1]).all():
        raise ValueError(f"Nonbinary candidate for {method}")
    base=_score(edges,gw)
    raw=_score(edges,candidate)
    # Retention is an explicit GW-seeded hybrid (applies to ALL methods).
    # Count the chance to keep the starting partition equally for everyone.
    if base > raw+1e-8:
        candidate=gw.copy()
        raw=base
        info["retained_gw_incumbent"]=True
    else:
        info["retained_gw_incumbent"]=False
    return candidate,raw,info


def _method_fingerprint(graphsha,factorsha,method,phase,cfg,direction_sha):
    manifest=dict(protocol="experiment024-paired-gw16-v1",
                  graph_sha256=graphsha,factor_sha256=factorsha,
                  method=method,phase=phase,cfg=cfg,
                  shared_directions_sha256=direction_sha)
    return hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:20]


BUDGETS={
    "pilot": dict(starts=16,seconds=120.,kmeans_iters=8,
        direction_trials=2,line_directions=2,em_stages=1,em_iters=2,
        cem_generations=2,cem_population=16,gibbs_sweeps=2,
        kl_passes=3,ils_cycles=2,ils_kl_passes=2),
    "full":dict(starts=16,seconds=300.,kmeans_iters=30,
        direction_trials=8,line_directions=8,em_stages=2,em_iters=4,
        cem_generations=4,cem_population=32,gibbs_sweeps=10,
        kl_passes=10,ils_cycles=16,ils_kl_passes=4)
}


def run_method(graph,method,phase,*,root=BASE):
    if graph not in GRAPH_IDS or method not in METHODS or phase not in BUDGETS:
        raise ValueError("Invalid graph, method or phase")
    began=time.monotonic()
    cfg=BUDGETS[phase].copy()
    folder=Path(root)/phase/graph
    factor_dir=select_factor(graph)
    graph_obj,y,_,graphsha,factorsha=_factor_and_case(Path("data")/(graph+".txt"),factor_dir)
    edges=edge_arrays(graph_obj)
    normals,gws,direct_meta=directions_archive(folder,y,graph,graphsha,factorsha)
    ident=_method_fingerprint(graphsha,factorsha,method,phase,cfg,direct_meta["direction_rows_sha256"])
    target=folder/method
    status_file=target/"status.json"
    if status_file.exists():
        s=json.loads(status_file.read_text(encoding="utf8"))
        if s.get("fingerprint")!=ident:
            raise ValueError("Incompatible method checkpoint (code, factor, or settings changed)")
        if s.get("status")=="complete_16" and (target/"raw16.npz").is_file():
            return s
    labels=[]
    scores=[]
    diagnostics=[]
    # All 16 directions are created once and reused verbatim by every method.
    for index,(r,gw) in enumerate(zip(normals,gws)):
        if time.monotonic()-began>cfg["seconds"]:
            break
        tick=time.monotonic()
        cand,score,diag=generate_candidate(
            graph_obj,y,edges,method,r,gw,index,cfg,folder,graphsha,factorsha)
        diag.update(index=index,raw_score=score,gw_cut=_score(edges,gw),
                    generation_seconds=time.monotonic()-tick)
        labels.append(cand)
        scores.append(score)
        diagnostics.append(diag)
    target.mkdir(parents=True,exist_ok=True)
    if labels:
        tmp=target/"raw16.tmp.npz"
        np.savez_compressed(tmp,raw_labels=np.stack(labels).astype(np.uint8),
                            raw_scores=np.asarray(scores,dtype=np.float64),
                            seed_indices=np.arange(len(scores),dtype=np.int16))
        os.replace(tmp,target/"raw16.npz")
    if len(scores)<16:
        out=dict(graph=graph,method=method,phase=phase,status="incomplete_starts",
                 starts_completed=len(scores),starts_requested=16,
                 fingerprint=ident,wall_seconds=time.monotonic()-began,
                 raw_best=max(scores) if scores else None,
                 no_primary_comparison=True)
        atomic_json(status_file,out)
        atomic_json(target/"diagnostics.json",diagnostics)
        return out
    winner=int(np.argmax(scores))
    raw=labels[winner]
    result,stages,details,partitions=refinement(
        graph_obj,edges,raw,method,winner,int(graph[1:]),cfg)
    if stages["Round"]!=scores[winner]:
        raise AssertionError("Saved raw best did not survive refinement entry")
    info=dict(graph=graph,method=method,phase=phase,
              fingerprint=ident,winning_seed_index=winner,
              gw_direction_index=winner,graph_sha256=graphsha,
              factor_sha256=factorsha,shared_direction_archive_sha256=sha(folder/"gw16_common.npz"),
              gw_selected_seed_cut=_score(edges,gws[winner]),
              method_notes=details,experiment="experiment024")
    write_case(folder,method,16,result,stages,info,partitions)
    if verify_case(folder,method,16,graph_obj,edges,ident) is None:
        raise AssertionError("Final partition failed audit")
    refs,_=load_references()
    out=dict(graph=graph,method=method,phase=phase,
             status="complete_16",starts_completed=16,starts_requested=16,
             selected_seed_index=winner,raw_best=float(scores[winner]),
             refined_best=float(stages["Final"]),
             raw_mean=float(np.mean(scores)),raw_std=float(np.std(scores,ddof=1)),
             gw_mean=float(np.mean([_score(edges,x) for x in gws])),
             gw_best=float(max(_score(edges,x) for x in gws)),
             full_common_refinement=True,method_specific_internal_evaluations_not_equal=True,
             method_family="GW-anchored-MST-hybrid" if method in MST else "GW-seeded",
             factor_sha256=factorsha,graph_sha256=graphsha,fingerprint=ident,
             reference_cut=refs[graph]["reference_cut"],
             reference_comparison=compare_cut(float(stages["Final"]),refs[graph]),
             wall_seconds=time.monotonic()-began,raw_archive_sha256=sha(target/"raw16.npz"))
    atomic_json(target/"diagnostics.json",diagnostics)
    atomic_json(status_file,out)
    return out


def run_batch(graph,phase,methods,*,root=BASE,timeout_seconds=460):
    if graph not in GRAPH_IDS:raise ValueError(graph)
    result=[]
    for method in methods:
        command=[sys.executable,"-m","maxkcut.experiment024_paired_gw16",
                 "single","--graph",graph,"--method",method,"--phase",phase,
                 "--root",str(root)]
        logdir=Path(root)/phase/graph/"logs"
        logdir.mkdir(parents=True,exist_ok=True)
        began=time.monotonic()
        try:
            proc=subprocess.run(command,timeout=timeout_seconds,check=True,
                 stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                 text=True,errors="replace",
                 env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
                      "NUMBA_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
            transcript=proc.stdout or ""
            state=json.loads((Path(root)/phase/graph/method/"status.json").read_text(encoding="utf8"))
        except subprocess.TimeoutExpired as e:
            transcript=(e.stdout or b"")
            if isinstance(transcript,bytes):transcript=transcript.decode("utf8","replace")
            state=dict(graph=graph,method=method,status="hard_timeout",phase=phase)
        except subprocess.CalledProcessError as e:
            transcript=e.stdout or ""
            state=dict(graph=graph,method=method,status="failed",phase=phase,exit_code=e.returncode)
        (logdir/(method+".log")).write_text(transcript[-25000:],encoding="utf8")
        state["orchestrator_seconds"]=time.monotonic()-began
        atomic_json(Path(root)/phase/graph/"statuses"/(method+".json"),state)
        result.append(state)
        print(json.dumps({k:state.get(k) for k in ("graph","method","status","starts_completed")}),flush=True)
    return result


def coverage(*,root=BASE,phase="full"):
    refs,_=load_references()
    rows=[]
    for graph in GRAPH_IDS:
        fam,weight_class=graph_family(graph)
        for method in METHODS:
            path=Path(root)/phase/graph/"statuses"/(method+".json")
            o=json.loads(path.read_text(encoding="utf8")) if path.exists() else {}
            rows.append(dict(graph=graph,family=fam,weight_class=weight_class,
                method=method,status=o.get("status","not_run"),
                starts_completed=o.get("starts_completed",0),
                raw_best=o.get("raw_best"),raw_mean=o.get("raw_mean"),
                refined_best=o.get("refined_best"),
                gw_best=o.get("gw_best"),
                best_known_reference=refs[graph]["reference_cut"],
                wall_seconds=o.get("wall_seconds")))
    dest=Path(root)/"analysis"
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/(phase+"_coverage.csv")).open("w",newline="",encoding="utf8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    good=sum(x["status"]=="complete_16" and int(x["starts_completed"])==16 for x in rows)
    out=dict(phase=phase,graphs=71,protocols=len(METHODS),
             planned_cells=71*len(METHODS),complete_16_cells=good,
             incomplete_cells=71*len(METHODS)-good,
             primary_analysis_valid=(good==71*len(METHODS)),
             statuses={s:sum(x["status"]==s for x in rows)
                       for s in sorted({x["status"] for x in rows})})
    atomic_json(dest/(phase+"_coverage.json"),out)
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=("single","batch","coverage"))
    ap.add_argument("--graph",choices=GRAPH_IDS,default="G11")
    ap.add_argument("--method",choices=METHODS,default="gw-rr")
    ap.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    ap.add_argument("--phase",choices=tuple(BUDGETS),default="pilot")
    ap.add_argument("--root",type=Path,default=BASE)
    ap.add_argument("--hard-timeout-seconds",type=float,default=460)
    args=ap.parse_args()
    if args.action=="single":response=run_method(args.graph,args.method,args.phase,root=args.root)
    elif args.action=="batch":response=run_batch(args.graph,args.phase,args.methods,
                               root=args.root,timeout_seconds=args.hard_timeout_seconds)
    else:response=coverage(root=args.root,phase=args.phase)
    print(json.dumps(response,indent=2,default=str))

if __name__=="__main__":main()
