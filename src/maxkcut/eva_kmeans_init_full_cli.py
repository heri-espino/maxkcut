"""Experiment019: matched full KMeansNM initialization -> Flip -> KL -> ILS.

All nine initializers share the ORIGINAL G63 signed graph and fixed
full EVA Y, restart IDs, the same spherical Lloyd iterations and
the *same refinement budget* used by the prior 26917 benchmark:
one-vertex positive-gain flips (20 sweeps), indexed generalized KL
(30 passes), and 80 cycles of ILS-KL with 8-pass KL inner steps.

This follows exactly the optimization stages/seeding of
eva_pipeline_grid.evaluate_pipeline, except that different two-center
initializers are permitted in the existing fixed-EVA clustering stage.

Stages are kept separately as independently verifiable binary cut
archives. Fully completed jobs are re-audited and skipped. Every
original signed-graph Max-Cut score is computed from the graph and
saved partitions. Original EVA remains immutable.
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
import traceback

import numpy as np

from .archive import (
    _file_sha256, load_method, new_archive, new_run_id,
    save_method, update_manifest, verify_archive,
)
from .eva_cli import _write_json, _git_push
from .eva_rounding_cli import _factor_and_case, _immutable_factor_copy
from .eva_multigroup_kmeans import optimize_multigroup_kmeans
from .eva_kmeans_initializers import INIT2
from .eva_elite_search import optimize_elite, optimize_kl_fast
from .io import edge_arrays
from .local_search import improve_by_flips
from .objective import cut_weight
from .sweep import discover_graphs

STAGES=("Round","Flip","KL","Final")
_G=None
_Y=None
_EDGES=None
_HASHES=None


def _init_worker(graph_path,source_dir,expected_hashes):
    global _G,_Y,_EDGES,_HASHES
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    g,y,_,graph_sha,factor_sha=_factor_and_case(
        Path(graph_path),Path(source_dir)
    )
    if [graph_sha,factor_sha]!=expected_hashes:
        raise RuntimeError("Worker original G63 graph/EVA changed")
    _G,_Y,_EDGES,_HASHES=g,y,edge_arrays(g),expected_hashes


def _fingerprint(init,indices,*,graph_sha,factor_sha,budget):
    settings={
        "version":"experiment019-kmeans-init-full-pipeline-v1",
        "initializer":init,"restart_ids":[int(i) for i in indices],
        "original_graph_sha256":graph_sha,
        "original_EVA_factor_sha256":factor_sha,
        "budget":budget,
        "stages":list(STAGES),
    }
    key=hashlib.sha256(
        json.dumps(settings,sort_keys=True,allow_nan=False).encode("utf8")
    ).hexdigest()[:20]
    return key,settings


def _verify(record,settings):
    if record.get("status")!="complete" or record.get("settings")!=settings:
        raise ValueError("Experiment019 resume has different settings")
    archive=Path(record["archive_path"])
    verified=verify_archive(archive)
    n=len(settings["restart_ids"])
    if (verified["status"]!="complete" or
            verified["graph_sha256"]!=settings["original_graph_sha256"]
            or len(record["trials"])!=n):
        raise ValueError("Experiment019 original signed graph archive mismatch")
    values={}
    for stage in STAGES:
        name=f"Exp019-{stage}_EVA"
        if (name not in verified["methods"] or
                verified["methods"][name]["verified_partitions"]!=n):
            raise ValueError(f"Experiment019 missing verified stage {stage}")
        parts,scores=load_method(archive/f"{name}.npz")
        values[stage]=(parts,scores)
    for i,trial in enumerate(record["trials"]):
        if (trial["restart"]!=settings["restart_ids"][i]
                or trial["initializer"]!=settings["initializer"]):
            raise ValueError("Experiment019 saved restart identity mismatch")
        previous=float("-inf")
        for stage in STAGES:
            observed=float(values[stage][1][i])
            if not np.isclose(
                trial["scores"][stage],observed,atol=1e-8,rtol=0
            ) or observed+1e-8<previous:
                raise ValueError("Experiment019 scores not verified/monotone")
            previous=observed
        if (trial["rounding"]["init"]!=settings["initializer"]
                or trial["rounding"]["k"]!=2):
            raise ValueError("Experiment019 clustering diagnostics changed")
        if trial["ils"]["cycles_completed"]>settings["budget"]["ils_cycles"]:
            raise ValueError("Saved ILS cycles exceed configured budget")
        if (trial["rounding"]["initialization_details"][
                "screened_hard_cuts"]!=
                (16 if settings["initializer"]=="gw-screen16" else 0)):
            raise ValueError("Experiment019 initial evaluation-cost mismatch")
    if not np.isclose(record["best_final"],max(
        x["scores"]["Final"] for x in record["trials"]
    ),atol=1e-8,rtol=0):
        raise ValueError("Experiment019 best final score mismatch")
    return record


def _run_chunk(job):
    config=job["settings"]
    case=Path(job["case_path"])
    if case.is_file():
        return _verify(
            json.loads(case.read_text(encoding="utf8")),config
        ),True
    if _G is None or _Y is None or _HASHES!=[
        config["original_graph_sha256"],
        config["original_EVA_factor_sha256"],
    ]:
        raise RuntimeError("Original G63 source not present in initialized worker")
    name=config["initializer"]
    b=config["budget"]
    graph_number=int(Path(job["graph_path"]).stem[1:])
    partitions={stage:[] for stage in STAGES}
    scores={stage:[] for stage in STAGES}
    trials=[]
    for restart in config["restart_ids"]:
        t0=time.monotonic()
        # Match the exact original Experiment008 KMeansNM restart stream.
        # Nonpaired strategies use shared independent-center seed
        # convention from Experiment018 for a matched restart comparison.
        rng=np.random.default_rng(np.random.SeedSequence([
            b["rounding_seed"],graph_number,2,restart,
            0 if name=="paired" else 1,
        ]))
        start=time.monotonic()
        candidates,rounding=optimize_multigroup_kmeans(
            _Y,_G,rng,k=2,init=name,
            max_iter=b["kmeans_iters"],top_merges=1,
        )
        raw=np.asarray(candidates[0]["labels"],dtype=np.int8).copy()
        timing={"round_seconds":time.monotonic()-start}
        s0=float(cut_weight(_EDGES,raw))
        # Original Experiment013 flip RNG exactly, *not* a
        # method-dependent seed that obscures initializer effects.
        frng=np.random.default_rng(np.random.SeedSequence([
            b["search_seed"],graph_number,restart,1313,
        ]))
        start=time.monotonic()
        flipped,flip_info=improve_by_flips(
            _G,raw.copy(),frng,max_sweeps=b["flip_sweeps"],
        )
        timing["flip_seconds"]=time.monotonic()-start
        s1=float(cut_weight(_EDGES,flipped))
        if s1+1e-8<s0:
            raise AssertionError("Positive-gain flip stage degraded cut")
        krng=np.random.default_rng(np.random.SeedSequence([
            b["search_seed"],graph_number,restart,101,
        ]))
        start=time.monotonic()
        kl,kl_info=optimize_kl_fast(
            _G,flipped,krng,passes=b["kl_passes"],
        )
        timing["kl_seconds"]=time.monotonic()-start
        s2=float(cut_weight(_EDGES,kl))
        if s2+1e-8<s1:
            raise AssertionError("Generalized KL degraded signed cut")
        irng=np.random.default_rng(np.random.SeedSequence([
            b["search_seed"],graph_number,restart,1401,
        ]))
        start=time.monotonic()
        final,ils_info=optimize_elite(
            _G,kl,[(kl,s2)],irng,
            method="ILS-KL",cycles=b["ils_cycles"],
            kl_passes=b["ils_kl_passes"],
            kick_sizes=tuple(min(len(kl),int(k))
                             for k in b["kick_sizes"]),
            temperature=b["temperature"],target=b["target"],
        )
        timing["ils_seconds"]=time.monotonic()-start
        timing["total_seconds"]=time.monotonic()-t0
        s3=float(cut_weight(_EDGES,final))
        if s3+1e-8<s2:
            raise AssertionError("ILS lost best incumbent")
        score_map=dict(zip(STAGES,(s0,s1,s2,s3)))
        for stage,labels in zip(STAGES,(raw,flipped,kl,final)):
            partitions[stage].append(np.asarray(labels,dtype=np.int8).copy())
            scores[stage].append(score_map[stage])
        trials.append({
            "restart":int(restart),"initializer":name,
            "scores":score_map,
            "rounding":rounding,
            "flip":flip_info,"kl":kl_info,"ils":ils_info,
            "timing":timing,
        })
    archive=new_archive(
        Path(job["output_dir"])/"archives",new_run_id(),
        Path(job["graph_path"]),_G.shape[0],_EDGES,2,
        config,jsonl_output=Path(job["trials_path"]),
        argv=["experiment019","initializer="+name],
    )
    for stage in STAGES:
        save_method(
            archive,f"Exp019-{stage}_EVA",
            partitions[stage],scores[stage],k=2,
        )
    update_manifest(archive,status="complete",solver_info={
        "original_EVA_factor_frozen":True,
        "original_graph_signed":True,
        "initializer":name,
        "rounding":"spherical-2-means",
        "postprocessing":"positive-gain flips -> generalized KL -> ILS-KL",
        "ils_implementation":"eva_elite_search.optimize_elite",
        "matched_paired_restart_seeds":True,
    })
    report=verify_archive(archive["path"])
    if (report["status"]!="complete"
            or report["graph_sha256"]!=config["original_graph_sha256"]
            or any(
                report["methods"][f"Exp019-{stage}_EVA"][
                    "verified_partitions"]!=len(trials)
                for stage in STAGES
            )):
        raise RuntimeError("New G63 full-pipeline archive does not verify")
    trial_path=Path(job["trials_path"])
    trial_path.parent.mkdir(parents=True,exist_ok=True)
    temp=trial_path.with_suffix(".tmp")
    with temp.open("w",encoding="utf8") as out:
        for row in trials:
            out.write(json.dumps({
                "graph":Path(job["graph_path"]).stem,
                "case_id":job["id"],**row,
            },sort_keys=True,allow_nan=False)+"\n")
    os.replace(temp,trial_path)
    # Experiment020 runs on ~68 graphs x 450 restarts. Keep each
    # complete 80-cycle ILS trace ONCE in the Git-backed JSONL,
    # while recording compact restart-level diagnostics in cases.
    # This avoids duplicating gigabytes of trajectory JSON without
    # sacrificing a single restart, stage score, bit-packed solution,
    # seed identity or recoverable trajectory.
    compact_trials=[]
    for trial in trials:
        compact=dict(trial)
        compact["rounding"]={
            k:v for k,v in trial["rounding"].items()
            if k!="trace"
        }
        compact["ils"]={
            k:v for k,v in trial["ils"].items()
            if k!="trace"
        }
        compact_trials.append(compact)
    record={
        "status":"complete","graph":Path(job["graph_path"]).stem,
        "id":job["id"],"settings":config,
        "archive_path":str(archive["path"]),
        "trials_path":str(trial_path),
        "best_final":float(max(scores["Final"])),
        "trials":compact_trials,
    }
    _write_json(case,record)
    return _verify(record,config),False


def _summary(dest,graph_name,target):
    dest=Path(dest)
    rows=[]
    for file in sorted((dest/"cases").glob(
        f"{graph_name}-init-full-*.json"
    )):
        doc=json.loads(file.read_text(encoding="utf8"))
        for trial in doc["trials"]:
            s=trial["scores"]
            rows.append({
                "graph":graph_name,
                "initializer":doc["settings"]["initializer"],
                "restart":trial["restart"],
                "raw_cut":s["Round"],
                "flip_cut":s["Flip"],
                "kl_cut":s["KL"],
                "final_cut":s["Final"],
                "improvement":s["Final"]-s["Round"],
                "gap_to_target":target-s["Final"],
                "extra_start_cuts":trial["rounding"][
                    "initialization_details"]["screened_hard_cuts"],
                "lloyd_iterations":trial["rounding"][
                    "lloyd_iterations"],
                "ils_cycles":trial["ils"]["cycles_completed"],
                "seconds":trial["timing"]["total_seconds"],
                "archive_path":doc["archive_path"],
                "case_id":doc["id"],
            })
    columns=[
        "graph","initializer","restart","raw_cut","flip_cut",
        "kl_cut","final_cut","improvement","gap_to_target",
        "extra_start_cuts","lloyd_iterations","ils_cycles",
        "seconds","archive_path","case_id",
    ]
    csv_path=dest/"summary.csv"
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    temp=csv_path.with_suffix(".tmp")
    with temp.open("w",encoding="utf8",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=columns)
        writer.writeheader()
        writer.writerows(sorted(
            rows,key=lambda x:(x["initializer"],x["restart"]),
        ))
    os.replace(temp,csv_path)
    grouped={}
    for row in rows:
        grouped.setdefault(row["initializer"],[]).append(row)
    ranks=[]
    for init,subset in grouped.items():
        arr=np.asarray([x["final_cut"] for x in subset],dtype=float)
        raw=np.asarray([x["raw_cut"] for x in subset],dtype=float)
        flips=np.asarray([x["flip_cut"] for x in subset],dtype=float)
        kl=np.asarray([x["kl_cut"] for x in subset],dtype=float)
        seconds=np.asarray([x["seconds"] for x in subset],dtype=float)
        best_idx=int(np.argmax(arr))
        ranks.append({
            "initializer":init,
            "restarts":len(subset),
            "best_restart":subset[best_idx]["restart"],
            "best_raw":float(raw.max()),
            "mean_raw":float(raw.mean()),
            "best_flip":float(flips.max()),
            "best_kl":float(kl.max()),
            "best_final":float(arr.max()),
            "mean_final":float(arr.mean()),
            "median_final":float(np.median(arr)),
            "mean_seconds":float(seconds.mean()),
            "extra_start_cuts_per_restart":subset[0]["extra_start_cuts"],
            "gap_to_target":float(target-arr.max()),
            "delta_vs_26917":float(arr.max()-26917),
        })
    ranks.sort(key=lambda r:(
        -r["best_final"],-r["mean_final"],r["initializer"],
    ))
    for idx,rank in enumerate(ranks,1):
        rank["rank"]=idx
    leaderboard=dest/"leaderboard.json"
    _write_json(leaderboard,{
        "graph":graph_name,"target":target,
        "historical_best":26917,
        "best_observed_this_experiment":(
            max((r["best_final"] for r in ranks),
                default=None)
        ),
        "leaderboard":ranks,
    })
    return csv_path,leaderboard,ranks


def run_graph(
    graph_path,*,factor_dir,dest,
    methods=INIT2,restarts=50,start_restart=0,
    chunk_size=5,workers=4,rounding_seed=2026,
    search_seed=2026,kmeans_iters=100,flip_sweeps=20,
    kl_passes=30,ils_cycles=80,ils_kl_passes=8,
    kick_sizes=(8,16,32,64),temperature=12.,
    target=27047.,
):
    graph_path,factor_dir,dest=map(Path,(
        graph_path,factor_dir,dest
    ))
    methods=tuple(methods)
    if (not methods or len(set(methods))!=len(methods)
            or any(x not in INIT2 for x in methods)
            or restarts<1 or start_restart<0 or chunk_size<1
            or workers<1 or rounding_seed<0 or search_seed<0
            or kmeans_iters<1 or flip_sweeps<1
            or kl_passes<1 or ils_cycles<1 or ils_kl_passes<1
            or len(kick_sizes)<1
            or any(int(k)!=k or k<1 for k in kick_sizes)
            or not np.isfinite(temperature) or temperature<0
            or not np.isfinite(target)):
        raise ValueError("Invalid Experiment019 initializer/search budget")
    graph,y,_,graph_sha,factor_sha=_factor_and_case(
        graph_path,factor_dir,
    )
    if any(k>len(y) for k in kick_sizes):
        raise ValueError("Kick size exceeds original graph vertex count")
    if dest.resolve()==factor_dir.resolve():
        raise ValueError("Refuse to overwrite source EVA factor folder")
    dest.mkdir(parents=True,exist_ok=True)
    _immutable_factor_copy(
        factor_dir,dest,graph_path.stem,factor_sha,
    )
    b={
        "rounding_seed":rounding_seed,
        "search_seed":search_seed,
        "kmeans_iters":kmeans_iters,
        "flip_sweeps":flip_sweeps,
        "kl_passes":kl_passes,
        "ils_cycles":ils_cycles,
        "ils_kl_passes":ils_kl_passes,
        "kick_sizes":list(kick_sizes),
        "temperature":temperature,
        "target":target,
    }
    jobs=[]
    for method in methods:
        for first in range(
            start_restart,start_restart+restarts,chunk_size
        ):
            ids=list(range(first,min(
                first+chunk_size,start_restart+restarts
            )))
            key,settings=_fingerprint(
                method,ids,graph_sha=graph_sha,
                factor_sha=factor_sha,budget=b,
            )
            base=f"{graph_path.stem}-init-full-{method}-{key}"
            jobs.append({
                "id":key,"settings":settings,
                "graph_path":str(graph_path),
                "output_dir":str(dest),
                "case_path":str(dest/"cases"/f"{base}.json"),
                "trials_path":str(dest/"trials"/f"{base}.jsonl"),
            })
    print(
        f"{graph_path.stem}: frozen EVA Y={y.shape}; "
        f"{len(methods)} initializer methods x {restarts} paired IDs "
        f"= {len(methods)*restarts} complete "
        f"Round->Flip->KL->ILS searches; "
        f"jobs={len(jobs)}; workers={workers}; "
        f"historical best=26917; target={target:g}.",
        flush=True,
    )
    done=[]
    errors=[]
    best=float("-inf")
    with ProcessPoolExecutor(
        max_workers=workers,initializer=_init_worker,
        initargs=(
            str(graph_path),str(factor_dir),
            [graph_sha,factor_sha],
        ),
    ) as pool:
        future_to_job={
            pool.submit(_run_chunk,job):job for job in jobs
        }
        for future in as_completed(future_to_job):
            job=future_to_job[future]
            try:
                record,was_cached=future.result()
                done.append(record)
                best=max(best,record["best_final"])
                print(
                    f"{graph_path.stem} {record['settings']['initializer']} "
                    f"{'verified skip' if was_cached else 'complete'} "
                    f"n={len(record['trials'])}; "
                    f"chunk best={record['best_final']:g}; "
                    f"best so far={best:g}; "
                    f"vs previous 26917={best-26917:+g}; "
                    f"gap to target={target-best:+g}",
                    flush=True,
                )
            except Exception as exc:
                issue={
                    "case":job["case_path"],
                    "error":f"{type(exc).__name__}: {exc}",
                    "traceback":traceback.format_exc().replace(
                        str(Path.home()),"<HOME>"
                    ),
                }
                errors.append(issue)
                _write_json(
                    dest/"errors"/(
                        Path(job["case_path"]).stem+".json"
                    ),
                    issue,
                )
                print(
                    f"{graph_path.stem} Experiment019 FAILED "
                    f"{job['settings']['initializer']}: {exc}",
                    flush=True,
                )
    # Windows parallel workers publish their atomic case JSON files
    # concurrently. Do NOT scan cases/ in the as_completed() loop:
    # a concurrent os.replace can temporarily deny opening a case file
    # (WinError 13). All futures are now joined before any summary
    # reads, so the entire graph ledger reflects a stable set of cases.
    summary,leaderboard,rank=_summary(
        dest,graph_path.stem,target,
    )
    if errors:
        raise RuntimeError(
            f"{len(errors)} Experiment019 chunks failed: "
            f"see {dest/'errors'}; completed case archives are preserved"
        )
    print(
        f"{graph_path.stem} Experiment019 COMPLETE: "
        f"best={best:g}, historical=26917, "
        f"gap to target={target-best:+g}; "
        f"summary={summary}; leaderboard={leaderboard}",
        flush=True,
    )
    return done,rank


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graphs",nargs="+",default=["G63"])
    p.add_argument("--gset-dir",type=Path,default=Path("data"))
    p.add_argument("--factor-dir",type=Path,default=Path(
        "results/gset/experiment003-eva-numba",
    ))
    p.add_argument("--output-dir",type=Path,default=Path(
        "results/gset/experiment019-kmeans-init-full-pipeline",
    ))
    p.add_argument("--methods",nargs="+",choices=INIT2,default=list(INIT2))
    p.add_argument("--restarts",type=int,default=50)
    p.add_argument("--start-restart",type=int,default=0)
    p.add_argument("--chunk-size",type=int,default=5)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--rounding-seed",type=int,default=2026)
    p.add_argument("--search-seed",type=int,default=2026)
    p.add_argument("--kmeans-iters",type=int,default=100)
    p.add_argument("--flip-sweeps",type=int,default=20)
    p.add_argument("--kl-passes",type=int,default=30)
    p.add_argument("--ils-cycles",type=int,default=80)
    p.add_argument("--ils-kl-passes",type=int,default=8)
    p.add_argument("--kick-sizes",nargs="+",type=int,
                   default=[8,16,32,64])
    p.add_argument("--temperature",type=float,default=12.)
    p.add_argument("--target",type=float,default=27047.)
    p.add_argument("--push",action="store_true")
    p.add_argument("--push-branch")
    opts=p.parse_args()
    if opts.push and not opts.push_branch:
        p.error("--push requires --push-branch")
    for graph_path in discover_graphs(
        opts.gset_dir,opts.graphs
    ):
        run_graph(
            graph_path,factor_dir=opts.factor_dir,
            dest=opts.output_dir,methods=opts.methods,
            restarts=opts.restarts,
            start_restart=opts.start_restart,
            chunk_size=opts.chunk_size,workers=opts.workers,
            rounding_seed=opts.rounding_seed,
            search_seed=opts.search_seed,
            kmeans_iters=opts.kmeans_iters,
            flip_sweeps=opts.flip_sweeps,
            kl_passes=opts.kl_passes,ils_cycles=opts.ils_cycles,
            ils_kl_passes=opts.ils_kl_passes,
            kick_sizes=tuple(opts.kick_sizes),
            temperature=opts.temperature,target=opts.target,
        )
    if opts.push:
        _git_push(
            opts.output_dir,
            "experiment019 KMeans initialization complete pipelines",
            allowed_branch=opts.push_branch,
        )


if __name__=="__main__":
    main()
