"""Experiment029: extend experiment028's p=41 vs p=n EVA/K-MeansNM comparison to 100 seeds.

Only missing EVA seeds 2031..2125 are optimized. All existing 2026..2030
factors are strictly checksum-verified and reused, never overwritten.
Distinct result directories, original signed-edge scores, restartable
Numba EVA checkpoints, original hard labels, and per-sweep traces are retained.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from maxkcut.eva import load_eva_factor, vector_objective
from maxkcut.eva_numba import solve_eva_numba
from maxkcut.io import load_graph, edge_arrays
from maxkcut.objective import cut_weight
from maxkcut.rounding import round_spherical_kmeans

GRAPHS=("G1","G6","G11","G14","G18")
OLD_SEEDS=tuple(range(2026,2031))
NEW_SEEDS=tuple(range(2031,2126))
ALL_SEEDS=OLD_SEEDS+NEW_SEEDS
RANKS=("default","full")
METHOD="K-MeansNM_EVA"
ROOT=Path("results/gset/experiment029-eva-kmeansnm-100seed")
EX028=Path("results/gset/experiment028-eva-fullrank-pilot")
FEATURES=Path("results/gset/experiment022-graph-structure-methods/graph_features.csv")
FIELDS=("graph","seed","rank_mode","p","n","m","family","weight_type", "graph_sha256",
        "source_experiment","factor_sha256","partition_sha256","cut_weight",
        "continuous_objective","eva_seconds","sweeps","converged","max_angle",
        "rounding_seconds","cluster_iterations","case_path","factor_path","partition_path")


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1024*1024),b""):
            h.update(part)
    return h.hexdigest()


def graph_hashes(path:Path)->dict:
    data=path.read_bytes();lf=data.replace(b"\r\n",b"\n")
    return {"raw":hashlib.sha256(data).hexdigest(),"lf":hashlib.sha256(lf).hexdigest(),
            "crlf":hashlib.sha256(lf.replace(b"\n",b"\r\n")).hexdigest()}


def properties(features:Path=FEATURES)->dict:
    with features.open(newline="",encoding="utf8") as f:
        return {r["graph"]:r for r in csv.DictReader(f)}


def load_verified_graph(graph:str,feature_map:dict,graph_dir:Path=Path("data"),allow_synthetic=False):
    path=graph_dir/f"{graph}.txt"
    a=load_graph(path)
    n=int(a.shape[0]);metadata=dict(feature_map[graph])
    if not allow_synthetic and n!=800:raise ValueError(f"Expected n=800 for {graph}: {n}")
    if n!=int(metadata["n"]):raise ValueError(f"Metadata vertex count mismatch for {graph}")
    hashes=graph_hashes(path)
    matches=[k for k,v in hashes.items() if v==metadata["graph_sha256"]]
    if not matches:raise ValueError(f"Source-graph SHA mismatch for {graph}: {hashes}")
    metadata["raw_checkout_sha256"]=hashes["raw"]
    metadata["accepted_checkout_eol"]=matches[0]
    return a,edge_arrays(a),metadata


def exact_row(graph,seed,mode,p,n,m,metadata,source,case,fac,part,score,obj,sweeps,conv,angle,
              elapsed,rounding_seconds,iterations, factor_hash,part_hash):
    return dict(graph=graph,seed=seed,rank_mode=mode,p=p,n=n,m=m,
       family=metadata["family"],weight_type=metadata["weight_type"],
       graph_sha256=metadata["graph_sha256"],source_experiment=source,
       factor_sha256=factor_hash,partition_sha256=part_hash,
       cut_weight=float(score),continuous_objective=float(obj),eva_seconds=float(elapsed),
       sweeps=int(sweeps),converged=bool(conv),max_angle=float(angle),
       rounding_seconds=float(rounding_seconds),cluster_iterations=int(iterations),
       case_path=case.as_posix(),factor_path=fac.as_posix(),partition_path=part.as_posix())


def audit_existing(graph,seed,mode,a,edges,metadata,base:Path=EX028):
    directory=base/"runs"/graph/mode/f"seed-{seed}"
    case,factor,part=(directory/p for p in ("case.json","factor.npz","partitions.npz"))
    item=json.loads(case.read_text(encoding="utf8"))
    settings=item["settings"]
    n=a.shape[0];p=int(np.ceil(np.sqrt(2*(n+1)))) if mode=="default" else n
    if (settings["graph"]!=graph or settings["seed"]!=seed or settings["p"]!=p or
        settings["rank_mode"]!=mode or settings["graph_sha256"]!=metadata["graph_sha256"]):
        raise RuntimeError(f"Mismatching historical EVA case {case}")
    if sha256(factor)!=item["factor_sha256"] or sha256(part)!=item["partitions_sha256"]:
        raise RuntimeError(f"Historical archives changed: {directory}")
    orig=next(x for x in item["rows"] if x["method"]==METHOD)
    with np.load(part,allow_pickle=False) as packed:
        labels=np.unpackbits(packed[METHOD])[:n]
        dig=hashlib.sha256(packed[METHOD].tobytes()).hexdigest()
    if dig!=orig["partition_sha256"] or abs(cut_weight(edges,labels)-float(orig["cut_weight"]))>1e-9:
        raise RuntimeError(f"Historical signed cut audit failed {directory}")
    y,state=load_eva_factor(factor)
    if y.shape!=(n,p) or abs(vector_objective(a,y)-float(orig["continuous_objective"]))>1e-4:
        raise RuntimeError(f"Historical EVA factor objective mismatch {directory}")
    return exact_row(graph,seed,mode,p,n,len(edges[0]),metadata,"experiment028",case,factor,part,
                     orig["cut_weight"],orig["continuous_objective"],orig["sweeps"],orig["converged"],
                     orig["max_angle"],orig["eva_seconds"],orig["rounding_seconds"],orig["cluster_iterations"],
                     item["factor_sha256"],orig["partition_sha256"])


def rng_for(seed:int,graph:str):
    # Identical Stream recipe to Experiment028, with K-MeansNM method_index=2.
    return np.random.default_rng(np.random.SeedSequence([seed,int(graph[1:]),2,2,28]))


def audit_new(graph,seed,mode,a,edges,metadata,root:Path=ROOT):
    folder=root/"runs"/graph/mode/f"seed-{seed}"
    case,fac,part=[folder/x for x in ("case.json","factor.npz","partitions.npz")]
    obj=json.loads(case.read_text(encoding="utf8"))
    if obj["graph"]!=graph or obj["seed"]!=seed or obj["rank_mode"]!=mode:
        raise RuntimeError(f"Corrupt case identity: {case}")
    row=obj["row"]
    if (row["graph_sha256"]!=metadata["graph_sha256"] or
         sha256(fac)!=row["factor_sha256"] or sha256(part)!=obj["partitions_archive_sha256"]):
        raise RuntimeError(f"Changed graph/factor/partition archive: {case}")
    n=a.shape[0];p=int(np.ceil(np.sqrt(2*(n+1)))) if mode=="default" else n
    y,state=load_eva_factor(fac)
    if y.shape!=(n,p) or int(state["sweeps"])!=int(row["sweeps"]):
        raise RuntimeError(f"Incorrect factor metadata {fac}")
    if not np.isclose(vector_objective(a,y),row["continuous_objective"],rtol=1e-9,atol=1e-5):
        raise RuntimeError(f"Incorrect factor objective {fac}")
    with np.load(part,allow_pickle=False) as x:
        if x.files!=[METHOD]:raise RuntimeError("Unexpected partition archive contents")
        packed=x[METHOD]
        labels=np.unpackbits(packed)[:n]
        if hashlib.sha256(packed.tobytes()).hexdigest()!=row["partition_sha256"]:
            raise RuntimeError(f"Packed partition checksum mismatch {part}")
    if abs(cut_weight(edges,labels)-float(row["cut_weight"]))>1e-9:
        raise RuntimeError(f"Incorrect original signed-cut score {part}")
    return row


def one_new(graph,seed,mode,a,edges,metadata,root,max_sweeps,tolerance,checkpoint_every):
    n=a.shape[0];p=int(np.ceil(np.sqrt(2*(n+1)))) if mode=="default" else n
    folder=root/"runs"/graph/mode/f"seed-{seed}"
    folder.mkdir(parents=True,exist_ok=True)
    case,fac,part,trace=[folder/x for x in ("case.json","factor.npz","partitions.npz","eva_sweeps.jsonl")]
    if case.is_file():
        cached=json.loads(case.read_text(encoding="utf8"))
        if (cached.get("max_sweeps")!=max_sweeps or cached.get("tolerance")!=tolerance or
            cached.get("checkpoint_every")!=checkpoint_every):
            raise RuntimeError(f"Refuse incompatible completed checkpoint {case}")
        return audit_new(graph,seed,mode,a,edges,metadata,root)
    if fac.exists():
        _,st=load_eva_factor(fac)
        completed=int(st["sweeps"])
        if trace.exists():
            old=[x for x in trace.read_text(encoding="utf8").splitlines()
                 if x.strip() and int(json.loads(x)["sweep"])<=completed]
            trace.write_text("\n".join(old)+("\n" if old else ""),encoding="utf8")
    def callback(s):
        with trace.open("a",encoding="utf8",newline="\n") as f:
            f.write(json.dumps({"sweep":s["sweeps"],"objective":s["objective"],
                "max_angle":s["max_angle"],"converged":s["converged"],
                "elapsed_seconds":s["elapsed_seconds"]},sort_keys=True)+"\n")
    print(f"EVA {graph} seed={seed} rank={mode} p={p}",flush=True)
    y,st=solve_eva_numba(a,rank=p,seed=seed,max_sweeps=max_sweeps,
      tolerance=tolerance,checkpoint_path=fac,checkpoint_every=checkpoint_every,on_sweep=callback)
    val=vector_objective(a,y)
    if not np.isclose(val,st["objective"],atol=1e-5,rtol=1e-8):
        raise RuntimeError("EVA objective mismatch")
    t=time.perf_counter()
    labels,iterations=round_spherical_kmeans(y,2,rng_for(seed,graph),
        antipodal=True,edges=edges,keep_best=True)
    round_seconds=time.perf_counter()-t
    labels=np.asarray(labels,dtype=np.uint8)
    if labels.shape!=(n,) or not set(np.unique(labels)).issubset({0,1}):
        raise RuntimeError("Invalid rounded partition")
    packed=np.packbits(labels)
    score=float(cut_weight(edges,labels))
    np.savez_compressed(part,**{METHOD:packed})
    ph=hashlib.sha256(packed.tobytes()).hexdigest()
    row=exact_row(graph,seed,mode,p,n,len(edges[0]),metadata,"experiment029",case,fac,part,score,val,
        st["sweeps"],st["converged"],st["max_angle"],st["elapsed_seconds"],
        round_seconds,iterations,sha256(fac),ph)
    obj={"schema":"experiment029-kmeansnm-v1","graph":graph,"seed":seed,
         "rank_mode":mode,"p":p,"max_sweeps":max_sweeps,"tolerance":tolerance,
         "checkpoint_every":checkpoint_every,"partitions_archive_sha256":sha256(part),
         "row":row,"runtime":{"numpy":np.__version__,"python":sys.version.split()[0],
                                  "platform":platform.platform()}}
    tmp=case.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n",encoding="utf8",newline="\n")
    os.replace(tmp,case)
    return audit_new(graph,seed,mode,a,edges,metadata,root)


def graph_run(graph,old,new,root,base,feature_map,graph_dir,max_sweeps,tolerance,checkpoint_every,allow_synthetic):
    a,edges,metadata=load_verified_graph(graph,feature_map,graph_dir,allow_synthetic)
    out=[]
    for seed in old:
        for mode in RANKS:
            out.append(audit_existing(graph,seed,mode,a,edges,metadata,base))
    for seed in new:
        for mode in RANKS:
            out.append(one_new(graph,seed,mode,a,edges,metadata,root,max_sweeps,tolerance,checkpoint_every))
        print(f"COMPLETE {graph} seed={seed}",flush=True)
    return out


def summarize(rows,graphs,seeds,root,require_complete=True,bootstrap_reps=5000):
    target={(g,s,mode) for g in graphs for s in seeds for mode in RANKS}
    actual={(r["graph"],int(r["seed"]),r["rank_mode"]) for r in rows}
    if len(rows)!=len(actual) or (require_complete and actual!=target) or not actual.issubset(target):
        raise RuntimeError(f"Incorrect coverage: expected {len(target)}, found {len(actual)} / {len(actual & target)} valid")
    root.mkdir(parents=True,exist_ok=True)
    rows=sorted(rows,key=lambda r:(r["graph"],int(r["seed"]),r["rank_mode"]))
    def write_csv(path,fields,records):
        with path.open("w",newline="",encoding="utf8") as f:
            w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n")
            w.writeheader();w.writerows(records)
    write_csv(root/"all_runs.csv",FIELDS,rows)
    by={(r["graph"],int(r["seed"]),r["rank_mode"]):r for r in rows}
    paired=[];table=[];rng=np.random.default_rng(29029)
    for g in graphs:
        pairs=[]
        for s in seeds:
            d=by.get((g,s,"default"));f=by.get((g,s,"full"))
            if not d or not f:
                if require_complete:raise RuntimeError(f"Missing rank pair {g} {s}")
                continue
            diff=float(f["cut_weight"])-float(d["cut_weight"])
            paired.append(dict(graph=g,seed=s,default_cut=d["cut_weight"],full_cut=f["cut_weight"],
               delta_full_minus_default=diff,default_objective=d["continuous_objective"],
               full_objective=f["continuous_objective"],default_eva_seconds=d["eva_seconds"],
               full_eva_seconds=f["eva_seconds"]))
            pairs.append((d,f,diff))
        if not pairs:continue
        values=np.array([p[2] for p in pairs]);ad=np.array([float(p[0]["cut_weight"]) for p in pairs]);af=np.array([float(p[1]["cut_weight"]) for p in pairs]);
        boot=values[rng.integers(len(values),size=(bootstrap_reps,len(values)))].mean(axis=1)
        lo,hi=np.quantile(boot,[.025,.975])
        dtime=np.mean([float(p[0]["eva_seconds"]) for p in pairs]);ftime=np.mean([float(p[1]["eva_seconds"]) for p in pairs]);
        table.append({"graph":g,"n":len(pairs),"default_mean":float(ad.mean()),"full_mean":float(af.mean()),
           "mean_delta":float(values.mean()),"sd_delta":float(values.std(ddof=1)) if len(values)>1 else 0.0,
           "ci95_low":float(lo),"ci95_high":float(hi),"wins_full":int((values>0).sum()),
           "ties":int((values==0).sum()),"losses_full":int((values<0).sum()),
           "best_default":float(ad.max()),"best_full":float(af.max()),
           "mean_default_objective":float(np.mean([float(p[0]["continuous_objective"]) for p in pairs])),
           "mean_full_objective":float(np.mean([float(p[1]["continuous_objective"]) for p in pairs])),
           "mean_default_eva_seconds":float(dtime),"mean_full_eva_seconds":float(ftime),
           "full_vs_default_runtime_ratio":float(ftime/dtime) if dtime else None,
           "default_converged":sum(bool(p[0]["converged"]) for p in pairs),
           "full_converged":sum(bool(p[1]["converged"]) for p in pairs)})
    write_csv(root/"paired_differences.csv",("graph","seed","default_cut","full_cut",
        "delta_full_minus_default","default_objective","full_objective","default_eva_seconds","full_eva_seconds"),paired)
    if table:write_csv(root/"per_graph.csv",tuple(table[0]),table)
    summary={"schema":"experiment029-kmeansnm-100seeds-v1","graphs":list(graphs),"seeds":list(seeds),
      "method":METHOD,"rows_observed":len(rows),"rows_expected":len(target),
      "paired_observed":len(paired),"paired_expected":len(graphs)*len(seeds),
      "complete":actual==target,"reused_prior_factor_cases":sum(r["source_experiment"]=="experiment028" for r in rows),
      "new_factor_cases":sum(r["source_experiment"]=="experiment029" for r in rows),
      "bootstrap_replicates":bootstrap_reps,"bootstrap_seed":29029,
      "macro_percent":float(np.mean([100*t["mean_delta"]/abs(t["default_mean"]) for t in table])) if table else None}
    (root/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf8",newline="\n")
    lines=["# Experiment029 — Antipodal K-MeansNM with 100 independent EVA seeds", "",
      "All hard cuts are rescored on original signed G-set edges. Each seed provides",
      "one independent EVA initialization and one K-MeansNM clustering run per rank.",
      "The first five seeds reuse Experiment028 factor/partition archives without recomputation.",
      "Internal EVA convergence is not a certificate of SDP global optimality.","",
      f"Coverage: {len(rows)}/{len(target)} score records; {len(paired)}/{len(graphs)*len(seeds)} paired comparisons.","",
      "| Graph | Mean p=41 | Mean p=800 | Delta | 95% paired bootstrap CI | Wins/ties/losses | Best p=41 | Best p=800 |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t in table:
        lines.append(f'| {t["graph"]} | {t["default_mean"]:.2f} | {t["full_mean"]:.2f} | {t["mean_delta"]:+.2f} | [{t["ci95_low"]:+.2f}, {t["ci95_high"]:+.2f}] | {t["wins_full"]}/{t["ties"]}/{t["losses_full"]} | {t["best_default"]:.0f} | {t["best_full"]:.0f} |')
    if table:
        lines.extend(["",f'Equal-graph relative mean change: **{summary["macro_percent"]:+.4f}%**.',"",
        "## EVA objective and runtime","", "| Graph | Objective p=41 | Objective p=800 | EVA time p=41 (s) | EVA time p=800 (s) | Ratio |", "|---|---:|---:|---:|---:|---:|"])
        for t in table:
            lines.append(f'| {t["graph"]} | {t["mean_default_objective"]:.4f} | {t["mean_full_objective"]:.4f} | {t["mean_default_eva_seconds"]:.3f} | {t["mean_full_eva_seconds"]:.3f} | {t["full_vs_default_runtime_ratio"]:.1f}× |')
    lines.extend(["","**Uncertainty:** Confidence intervals resample 100 paired seed identifiers **within each graph**, conditioning on the five selected graphs. They do not quantify between-graph variation or the significance of a population-level rank advantage.","**Cost caveat:** Runtime excludes compilation overhead and artifact/IO time; branch and environment comparisons require identical hardware.","**Factor width is not effective numerical rank.** No SDP dual certificate is inferred.",""])
    (root/"report.md").write_text("\n".join(lines),encoding="utf8",newline="\n")
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graphs",nargs="+",default=list(GRAPHS))
    p.add_argument("--old-seeds",type=int,nargs="+",default=list(OLD_SEEDS))
    p.add_argument("--new-seeds",type=int,nargs="+",default=list(NEW_SEEDS))
    p.add_argument("--output",type=Path,default=ROOT)
    p.add_argument("--prior",type=Path,default=EX028)
    p.add_argument("--features",type=Path,default=FEATURES)
    p.add_argument("--graph-dir",type=Path,default=Path("data"))
    p.add_argument("--workers",type=int,default=5)
    p.add_argument("--max-sweeps",type=int,default=1000)
    p.add_argument("--tolerance",type=float,default=1e-3)
    p.add_argument("--checkpoint-every",type=int,default=25)
    p.add_argument("--validate-only",action="store_true")
    p.add_argument("--check-only",action="store_true")
    p.add_argument("--allow-synthetic",action="store_true")
    a=p.parse_args(argv)
    if len(set(a.graphs))!=len(a.graphs) or len(set(a.old_seeds+a.new_seeds))!=len(a.old_seeds+a.new_seeds):
        p.error("Graph and seed IDs must be unique")
    if not a.graphs or not a.old_seeds or not a.new_seeds or a.workers<1 or a.workers>5:
        p.error("Invalid graph/seed/worker argument")
    if not a.allow_synthetic and (tuple(a.graphs)!=GRAPHS or tuple(a.old_seeds)!=OLD_SEEDS or tuple(a.new_seeds)!=NEW_SEEDS):
        p.error("Experiment029 protocol is fixed; use separate branch/experiment for a changed design")
    if a.max_sweeps<1 or a.tolerance<=0 or a.checkpoint_every<1:
        p.error("Invalid EVA parameters")
    props=properties(a.features)
    for g in a.graphs:
        load_verified_graph(g,props,a.graph_dir,a.allow_synthetic)
        print(f"Input verified {g}",flush=True)
    if a.validate_only:
        print("PASS: graph edge lists, feature hashes and required graph sizes",flush=True)
        return
    if a.check_only:
        rows=[]
        for g in a.graphs:
            gr,edges,meta=load_verified_graph(g,props,a.graph_dir,a.allow_synthetic)
            for seed in a.old_seeds:
                for mode in RANKS:rows.append(audit_existing(g,seed,mode,gr,edges,meta,a.prior))
            for seed in a.new_seeds:
                for mode in RANKS:rows.append(audit_new(g,seed,mode,gr,edges,meta,a.output))
        print(json.dumps(summarize(rows,tuple(a.graphs),tuple(a.old_seeds+a.new_seeds),a.output),indent=2))
        return
    allrows=[]
    if a.workers==1:
        for g in a.graphs:
            allrows.extend(graph_run(g,a.old_seeds,a.new_seeds,a.output,a.prior,props,a.graph_dir,
                       a.max_sweeps,a.tolerance,a.checkpoint_every,a.allow_synthetic))
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as executor:
            jobs={executor.submit(graph_run,g,a.old_seeds,a.new_seeds,a.output,a.prior,props,a.graph_dir,
                  a.max_sweeps,a.tolerance,a.checkpoint_every,a.allow_synthetic):g for g in a.graphs}
            for job in as_completed(jobs):
                try: allrows.extend(job.result())
                except Exception as error:
                    for f in jobs:f.cancel()
                    raise RuntimeError(f"Graph worker failed: {jobs[job]}") from error
                summarize(allrows,tuple(a.graphs),tuple(a.old_seeds+a.new_seeds),a.output,require_complete=False)
    summary=summarize(allrows,tuple(a.graphs),tuple(a.old_seeds+a.new_seeds),a.output)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":main()