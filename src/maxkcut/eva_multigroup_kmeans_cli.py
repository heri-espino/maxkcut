"""Experiment017: evaluate all two-way merges of multi-group spherical KMeansNM.

Reuses ONLY the completed original EVA factor; never reruns EVA/SDP.
Each independent restart begins from k/2 antipodal centroid pairs
(k=2 is the EXACT original K-MeansNM baseline), or ordinary
independent k-center initialization as a declared control.

At every Lloyd iteration, evaluates all 2^(k-1) group assignments
(up to global complement) against original signed weighted Max-Cut.
Keep top distinct binary coarsenings ACROSS iterations, evaluate KL
from each, and archive all tested top-candidate binary partitions,
their full k-way group labels, integer group-merge masks, and all
signed original-graph cut scores. StageRaw is the highest raw group
merge; StageBeforeKL is the source of highest refined candidate;
StageAfterKL is the highest KL-refined cut.

Windows ProcessPoolExecutor with source factor/graph SHA checks,
settings-fingerprinted case archives and independent bit-level
verification. Repeated runs skip already verified completed cases.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np

from .archive import (
    _atomic_npz, _file_sha256, load_method, new_archive,
    new_run_id, save_method, update_manifest, verify_archive,
)
from .eva_cli import _write_json, _git_push
from .eva_rounding_cli import _factor_and_case, _immutable_factor_copy
from .eva_multigroup_kmeans import optimize_multigroup_kmeans
from .eva_elite_search import optimize_kl_fast
from .io import edge_arrays
from .objective import cut_weight
from .sweep import discover_graphs


STAGES=("Raw","BeforeKL","AfterKL")
EXTRA_STAGES=("CandidatesRaw","CandidatesKL")
INIT_MODES=("paired","independent")
_G=None
_Y=None
_EDGES=None
_SOURCE=None


def _init(path, factor_dir, hashes):
    global _G,_Y,_EDGES,_SOURCE
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    graph,y,_,graph_sha,factor_sha=_factor_and_case(
        Path(path),Path(factor_dir),
    )
    if [graph_sha,factor_sha]!=hashes:
        raise ValueError("Experiment017 original graph/EVA factor mismatch")
    _G,_Y,_EDGES=graph,y,edge_arrays(graph)
    _SOURCE=[graph_sha,factor_sha]


def _job_id(k, init, restart_ids, *, graph_sha, factor_sha, budget):
    config={
        "version":"experiment017-multigroup-coarsening-v1",
        "graph_sha256":graph_sha,
        "factor_sha256":factor_sha,
        "k":int(k),"init":init,
        "restart_ids":[int(r) for r in restart_ids],
        "budget":budget,
        "optimization":"exact coarse binary signed Max-Cut over k groups",
    }
    key=hashlib.sha256(
        json.dumps(config,sort_keys=True,allow_nan=False).encode("utf8")
    ).hexdigest()[:20]
    return key,config


def _verify(record,config):
    if record.get("status")!="complete" or record.get("config")!=config:
        raise ValueError("Cached multi-group case has different settings")
    report=verify_archive(record["archive_path"])
    n=len(config["restart_ids"])
    if (report["status"]!="complete"
            or report["graph_sha256"]!=config["graph_sha256"]
            or len(record["trials"])!=n):
        raise ValueError("Cached multi-group signed graph archive mismatch")
    archive=Path(record["archive_path"])
    stage_data={}
    for stage in STAGES+EXTRA_STAGES:
        name=f"Exp017-{stage}_EVA"
        if name not in report["methods"]:
            raise ValueError(f"Missing multigroup stage {stage}")
        labels,scores=load_method(archive/f"{name}.npz")
        count=(n if stage in STAGES else
               sum(len(x["candidates"]) for x in record["trials"]))
        if report["methods"][name]["verified_partitions"]!=count:
            raise ValueError(f"Cached {stage} stage count mismatch")
        stage_data[stage]=(labels,scores)
    group_file=Path(record["groups_path"])
    if _file_sha256(group_file)!=record["groups_sha256"]:
        raise ValueError("Original k-way clusters file SHA mismatch")
    with np.load(group_file,allow_pickle=False) as data:
        groups=np.asarray(data["groups"])
        masks=np.asarray(data["masks"])
        restarts=np.asarray(data["restarts"])
        iterations=np.asarray(data["iterations"])
    num_candidates=len(stage_data["CandidatesRaw"][1])
    if (groups.shape!=(_Y.shape[0],num_candidates)
            or masks.shape!=(num_candidates,)
            or restarts.shape!=(num_candidates,)
            or iterations.shape!=(num_candidates,)
            or np.any(groups<0) or np.any(groups>=config["k"])):
        raise ValueError("Original k-way group archive is invalid")
    binary,scores=stage_data["CandidatesRaw"]
    for j in range(num_candidates):
        # Group 0 is anchored to side zero, and each higher group
        # is assigned via one bit of the saved integer mask.
        predicted=(
            (masks[j] >> (groups[:,j].astype(np.int32)-1).clip(min=0)) & 1
        ).astype(np.int8)
        predicted[groups[:,j]==0]=0
        if not np.array_equal(binary[j],predicted):
            raise ValueError("Archived binary cut not generated by group merge")
    offset=0
    for idx,trial in enumerate(record["trials"]):
        if trial["restart"]!=config["restart_ids"][idx]:
            raise ValueError("Cached restart sequence differs")
        candidates=trial["candidates"]
        if not candidates:
            raise ValueError("Missing top group-merge candidates")
        for j,entry in enumerate(candidates):
            pos=offset+j
            if (int(restarts[pos])!=trial["restart"]
                    or int(masks[pos])!=entry["mask"]
                    or int(iterations[pos])!=entry["iteration"]
                    or not np.isclose(
                        scores[pos],entry["raw_cut"],rtol=0,atol=1e-8,
                    )
                    or not np.isclose(
                        stage_data["CandidatesKL"][1][pos],
                        entry["kl_cut"],rtol=0,atol=1e-8,
                    )):
                raise ValueError("Cached group candidate/score mismatch")
        for stage in STAGES:
            if not np.isclose(
                stage_data[stage][1][idx],trial["scores"][stage],
                rtol=0,atol=1e-8,
            ):
                raise ValueError("Cached best group-merge score mismatch")
        s=trial["scores"]
        if (s["AfterKL"]+1e-8<s["BeforeKL"]
                or s["AfterKL"]+1e-8<s["Raw"]
                or s["Raw"]+1e-8<
                    max(row["raw_cut"] for row in candidates)):
            raise ValueError("KL refinement or group enumeration lost the best")
        offset+=len(candidates)
    if offset!=num_candidates:
        raise ValueError("Candidate index count mismatch")
    return record


def _solve(job):
    config=job["config"]
    case=Path(job["case_path"])
    if case.is_file():
        return _verify(json.loads(
            case.read_text(encoding="utf8"),
        ),config),True
    if _G is None or _SOURCE!=[
        config["graph_sha256"],config["factor_sha256"],
    ]:
        raise RuntimeError("Multi-KMeans worker EVA source not verified")
    budget=config["budget"]
    graph_num=int(Path(job["graph_path"]).stem[1:])
    k=config["k"]
    init=config["init"]
    stages={stage:[] for stage in STAGES+EXTRA_STAGES}
    cuts={stage:[] for stage in STAGES+EXTRA_STAGES}
    all_groups,all_masks,all_restarts,all_iterations=[],[],[],[]
    trials=[]
    for restart in config["restart_ids"]:
        # For k=2, paired, this is EXACTLY the existing
        # K-MeansNM_EVA original rounding RNG convention.
        clustering_rng=np.random.default_rng(np.random.SeedSequence([
            budget["rounding_seed"],graph_num,2,restart,
            0 if init=="paired" else 1,
        ]))
        candidates,clustering=optimize_multigroup_kmeans(
            _Y,_G,clustering_rng,k=k,init=init,
            max_iter=budget["max_iter"],
            top_merges=budget["top_merges"],
        )
        winner=None
        evaluated=[]
        for rank,candidate in enumerate(candidates):
            raw=candidate["labels"]
            raw_cut=float(cut_weight(_EDGES,raw))
            if not np.isclose(
                raw_cut,candidate["score"],rtol=0,atol=1e-8,
            ):
                raise AssertionError("Multigroup raw signed cut inconsistent")
            kl_rng=np.random.default_rng(np.random.SeedSequence([
                budget["kl_seed"],graph_num,restart,101,
                k,0 if init=="paired" else 1,rank,
            ]))
            kl_labels,kl_report=optimize_kl_fast(
                _G,raw,kl_rng,passes=budget["kl_passes"],
            )
            kl_cut=float(cut_weight(_EDGES,kl_labels))
            if kl_cut+1e-8<raw_cut:
                raise AssertionError("KL lowered original signed Max-Cut")
            stages["CandidatesRaw"].append(raw.copy())
            stages["CandidatesKL"].append(kl_labels.copy())
            cuts["CandidatesRaw"].append(raw_cut)
            cuts["CandidatesKL"].append(kl_cut)
            all_groups.append(candidate["groups"].copy())
            all_masks.append(candidate["mask"])
            all_restarts.append(restart)
            all_iterations.append(candidate["iteration"])
            evaluated.append({
                "rank":rank,
                "mask":int(candidate["mask"]),
                "iteration":int(candidate["iteration"]),
                "raw_cut":raw_cut,"kl_cut":kl_cut,
                "kl_passes_improved":kl_report["passes_improved"],
                "kl_flips_retained":kl_report["flips_retained"],
            })
            if winner is None or kl_cut>winner["score"]+1e-8:
                winner={
                    "raw":raw.copy(),"refined":kl_labels.copy(),
                    "raw_score":raw_cut,"score":kl_cut,
                    "rank":rank,"mask":candidate["mask"],
                }
        assert winner is not None
        best_raw=candidates[0]
        raw_cut=float(best_raw["score"])
        selected_raw=winner["raw"]
        selected_raw_cut=winner["raw_score"]
        final=winner["refined"]
        final_cut=winner["score"]
        stages["Raw"].append(best_raw["labels"].copy())
        stages["BeforeKL"].append(selected_raw.copy())
        stages["AfterKL"].append(final.copy())
        cuts["Raw"].append(raw_cut)
        cuts["BeforeKL"].append(selected_raw_cut)
        cuts["AfterKL"].append(final_cut)
        if final_cut+1e-8<raw_cut:
            raise AssertionError("Best KL result lost best raw coarsening")
        trials.append({
            "restart":int(restart),"k":k,
            "init":init,
            "scores":{
                "Raw":raw_cut,
                "BeforeKL":selected_raw_cut,
                "AfterKL":final_cut,
            },
            "winner_rank":winner["rank"],
            "winner_merge_mask":int(winner["mask"]),
            "clustering":clustering,
            "candidates":evaluated,
        })
    group_path=Path(job["groups_path"])
    _atomic_npz(
        group_path,
        groups=np.stack(all_groups,axis=1).astype(np.uint8),
        masks=np.asarray(all_masks,dtype=np.uint32),
        restarts=np.asarray(all_restarts,dtype=np.int32),
        iterations=np.asarray(all_iterations,dtype=np.int16),
    )
    group_sha=_file_sha256(group_path)
    archive=new_archive(
        Path(job["output_dir"])/"archives",new_run_id(),
        Path(job["graph_path"]),_G.shape[0],_EDGES,2,config,
        jsonl_output=Path(job["trials_path"]),
        argv=["experiment017",f"k={k}",f"init={init}"],
    )
    for stage in STAGES+EXTRA_STAGES:
        save_method(
            archive,f"Exp017-{stage}_EVA",
            stages[stage],cuts[stage],k=2,
        )
    update_manifest(archive,status="complete",solver_info={
        "original_EVA_factor_frozen":True,
        "original_graph_signed":True,
        "maxcut_groups_exact":True,
        "initialization":init,"k":k,
        "groups_file":str(group_path),
        "groups_sha256":group_sha,
        "group_merges_per_iteration":int(1<<(k-1)),
        "KL_separate_from_raw_coarsening":True,
    })
    report=verify_archive(archive["path"])
    if (report["status"]!="complete"
            or report["graph_sha256"]!=config["graph_sha256"]
            or any(
                report["methods"][f"Exp017-{stage}_EVA"][
                    "verified_partitions"]!=len(stages[stage])
                for stage in STAGES+EXTRA_STAGES
            )):
        raise RuntimeError("New multigroup archive did not verify")
    trial_path=Path(job["trials_path"])
    trial_path.parent.mkdir(parents=True,exist_ok=True)
    temporary=trial_path.with_suffix(".tmp")
    with temporary.open("w",encoding="utf8") as handle:
        for row in trials:
            handle.write(json.dumps({
                "graph":Path(job["graph_path"]).stem,
                "configuration_id":job["id"],
                "archive_path":str(archive["path"]),
                **row,
            },allow_nan=False,sort_keys=True)+"\n")
    os.replace(temporary,trial_path)
    record={
        "status":"complete",
        "graph":Path(job["graph_path"]).stem,
        "id":job["id"],"config":config,
        "archive_path":str(archive["path"]),
        "groups_path":str(group_path),
        "groups_sha256":group_sha,
        "trials_path":str(trial_path),
        "trials":trials,
        "best_raw":float(max(cuts["Raw"])),
        "best_KL":float(max(cuts["AfterKL"])),
    }
    _write_json(case,record)
    return _verify(record,config),False


def write_summaries(dest,*,graph_name,target):
    dest=Path(dest)
    rows=[]
    for path in sorted((dest/"cases").glob(
        f"{graph_name}-multi-*.json",
    )):
        record=json.loads(path.read_text(encoding="utf8"))
        k=record["config"]["k"]
        init=record["config"]["init"]
        for trial in record["trials"]:
            s=trial["scores"]
            rows.append({
                "graph":graph_name,
                "k":k,"init":init,"restart":trial["restart"],
                "raw_cut":s["Raw"],"pre_KL_cut":s["BeforeKL"],
                "post_KL_cut":s["AfterKL"],
                "gain_from_KL":s["AfterKL"]-s["Raw"],
                "gap_to_target":float(target)-s["AfterKL"],
                "lloyd_iterations":trial["clustering"]["lloyd_iterations"],
                "merges_per_iteration":1<<(k-1),
                "top_candidates":len(trial["candidates"]),
                "winning_mask":trial["winner_merge_mask"],
                "archive_path":record["archive_path"],
                "groups_path":record["groups_path"],
                "case_id":record["id"],
            })
    columns=[
        "graph","k","init","restart",
        "raw_cut","pre_KL_cut","post_KL_cut","gain_from_KL",
        "gap_to_target","lloyd_iterations",
        "merges_per_iteration","top_candidates","winning_mask",
        "archive_path","groups_path","case_id",
    ]
    summary=dest/"summary.csv"
    summary.parent.mkdir(parents=True,exist_ok=True)
    temporary=summary.with_suffix(".tmp")
    with temporary.open("w",encoding="utf8",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=columns)
        writer.writeheader()
        writer.writerows(sorted(rows,key=lambda x:(
            x["k"],x["init"],x["restart"],
        )))
    os.replace(temporary,summary)
    grouped={}
    for row in rows:
        grouped.setdefault((row["k"],row["init"]),[]).append(row)
    leaderboard=[]
    for (k,init),items in grouped.items():
        raw=np.asarray([x["raw_cut"] for x in items],dtype=float)
        refined=np.asarray([x["post_KL_cut"] for x in items],dtype=float)
        idx=int(np.argmax(refined))
        leaderboard.append({
            "k":k,"init":init,"restarts":len(items),
            "best_raw":float(raw.max()),
            "mean_raw":float(raw.mean()),
            "best_KL":float(refined.max()),
            "mean_KL":float(refined.mean()),
            "best_restart":int(items[idx]["restart"]),
            "gap_to_target":float(target-refined.max()),
        })
    leaderboard.sort(key=lambda row:(
        -row["best_KL"],-row["mean_KL"],row["k"],row["init"],
    ))
    for i,row in enumerate(leaderboard,1):
        row["rank"]=i
    rank_file=dest/"leaderboard.json"
    _write_json(rank_file,{
        "graph":graph_name,"target":target,
        "leaderboard":leaderboard,
    })
    return summary,rank_file,leaderboard


def run_graph(
    path,*,factor_dir,dest,ks=(2,4,6,8,10),
    inits=("paired",),restarts=12,start_restart=0,
    chunk_size=3,workers=4,rounding_seed=2026,
    kl_seed=2026,max_iter=100,top_merges=4,
    kl_passes=30,target=27047.,
):
    path,factor_dir,dest=map(Path,(path,factor_dir,dest))
    ks=tuple(int(k) for k in ks)
    if (not ks or len(set(ks))!=len(ks)
            or not inits or len(set(inits))!=len(inits)
            or any(mode not in INIT_MODES for mode in inits)
            or any(k<2 or k>16 or
                   ("paired" in inits and k%2) for k in ks)
            or restarts<1 or start_restart<0 or chunk_size<1
            or workers<1 or max_iter<1 or top_merges<1
            or kl_passes<1 or rounding_seed<0 or kl_seed<0
            or not np.isfinite(target)):
        raise ValueError("Invalid Experiment017 k-grid/worker/seed budget")
    graph,y,_,graph_sha,factor_sha=_factor_and_case(path,factor_dir)
    if any(k>len(y) for k in ks):
        raise ValueError("Requested more spherical clusters than vertices")
    if dest.resolve()==factor_dir.resolve():
        raise ValueError("Refuse to overwrite original saved EVA factors")
    dest.mkdir(parents=True,exist_ok=True)
    _immutable_factor_copy(factor_dir,dest,path.stem,factor_sha)
    budget={
        "rounding_seed":rounding_seed,"kl_seed":kl_seed,
        "max_iter":max_iter,"top_merges":top_merges,
        "kl_passes":kl_passes,"target":float(target),
    }
    jobs=[]
    for k in ks:
        for mode in inits:
            for first in range(
                start_restart,start_restart+restarts,chunk_size,
            ):
                ids=list(range(first,min(
                    first+chunk_size,start_restart+restarts,
                )))
                key,config=_job_id(
                    k,mode,ids,graph_sha=graph_sha,
                    factor_sha=factor_sha,budget=budget,
                )
                name=f"{path.stem}-multi-k{k}-{mode}-{key}"
                jobs.append({
                    "id":key,"config":config,
                    "graph_path":str(path),"output_dir":str(dest),
                    "case_path":str(dest/"cases"/f"{name}.json"),
                    "groups_path":str(dest/"groups"/f"{name}.npz"),
                    "trials_path":str(dest/"trials"/f"{name}.jsonl"),
                })
    print(
        f"{path.stem}: verified immutable EVA Y={y.shape}; "
        f"ks={list(ks)}; inits={list(inits)}; "
        f"{restarts} paired seeds each, {len(jobs)} jobs, "
        f"workers={workers}; target={target:g}. "
        f"k=10 has 511 nontrivial group merges per Lloyd iteration.",
        flush=True,
    )
    failures=[]
    observed=[]
    global_best=float("-inf")
    with ProcessPoolExecutor(
        max_workers=workers,initializer=_init,
        initargs=(str(path),str(factor_dir),[graph_sha,factor_sha]),
    ) as pool:
        futures={pool.submit(_solve,job):job for job in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:
                record,cached=future.result()
                observed.append(record)
                global_best=max(global_best,record["best_KL"])
                info=record["config"]
                print(
                    f"{path.stem} k={info['k']} {info['init']} "
                    f"{'verified skip' if cached else 'complete'} "
                    f"n={len(record['trials'])}: "
                    f"raw best={record['best_raw']:g}; "
                    f"KL best={record['best_KL']:g}; "
                    f"global multigroup best={global_best:g} "
                    f"(gap to {target:g}: {target-global_best:+g})",
                    flush=True,
                )
            except Exception as exc:
                issue={
                    "case":job["case_path"],
                    "error":f"{type(exc).__name__}: {exc}",
                    "traceback":traceback.format_exc().replace(
                        str(Path.home()),"<HOME>",
                    ),
                }
                failures.append(issue)
                _write_json(
                    dest/"errors"/(
                        Path(job["case_path"]).stem+".json"
                    ),issue,
                )
                print(f"{path.stem} multigroup FAILED: {exc}",flush=True)
            write_summaries(dest,graph_name=path.stem,target=target)
    if failures:
        raise RuntimeError(
            f"{len(failures)} Experiment017 chunks failed; "
            f"see {dest/'errors'}"
        )
    summary,leaderboard,ranks=write_summaries(
        dest,graph_name=path.stem,target=target,
    )
    print(
        f"{path.stem} multigroup COMPLETE best={global_best:g}; "
        f"gap to {target:g} = {target-global_best:+g}; "
        f"summary={summary}; leaderboard={leaderboard}",
        flush=True,
    )
    return observed,ranks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs",nargs="+",default=["G63"])
    parser.add_argument("--gset-dir",type=Path,default=Path("data"))
    parser.add_argument("--factor-dir",type=Path,default=Path(
        "results/gset/experiment003-eva-numba",
    ))
    parser.add_argument("--output-dir",type=Path,default=Path(
        "results/gset/experiment017-multigroup-coarsening",
    ))
    parser.add_argument("--ks",nargs="+",type=int,
                        default=[2,4,6,8,10])
    parser.add_argument("--inits",nargs="+",choices=INIT_MODES,
                        default=["paired"])
    parser.add_argument("--restarts",type=int,default=12)
    parser.add_argument("--start-restart",type=int,default=0)
    parser.add_argument("--chunk-size",type=int,default=3)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--rounding-seed",type=int,default=2026)
    parser.add_argument("--kl-seed",type=int,default=2026)
    parser.add_argument("--max-iter",type=int,default=100)
    parser.add_argument("--top-merges",type=int,default=4)
    parser.add_argument("--kl-passes",type=int,default=30)
    parser.add_argument("--target",type=float,default=27047.)
    parser.add_argument("--push",action="store_true")
    parser.add_argument("--push-branch")
    args=parser.parse_args()
    if args.push and not args.push_branch:
        parser.error("--push requires --push-branch")
    paths=discover_graphs(args.gset_dir,args.graphs)
    for path in paths:
        run_graph(
            path,factor_dir=args.factor_dir,dest=args.output_dir,
            ks=args.ks,inits=tuple(args.inits),restarts=args.restarts,
            start_restart=args.start_restart,
            chunk_size=args.chunk_size,workers=args.workers,
            rounding_seed=args.rounding_seed,kl_seed=args.kl_seed,
            max_iter=args.max_iter,top_merges=args.top_merges,
            kl_passes=args.kl_passes,target=args.target,
        )
    if args.push:
        _git_push(
            args.output_dir,
            "experiment017 k-group exact signed coarsening",
            allowed_branch=args.push_branch,
        )


if __name__=="__main__":
    main()
