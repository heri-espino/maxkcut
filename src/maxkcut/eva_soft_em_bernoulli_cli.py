"""Experiment016 CLI: fixed EVA, multi-seed soft EM, CEM, Gibbs, KL.

Reuses the original completed EVA Y and original signed G-set graph,
independently verifies high-scoring Experiment013 archive partitions
when requested, and never changes the factor. Compares:
  EM+KL, EM+CEM, EM+Gibbs, EM+CEM+KL
on mean-direction and elite warm starts with paired deterministic RNGs.

Save EM Bernoulli probabilities, CEM search probabilities and four
full sets of bit-packed hard partitions: Source, EMHard, Candidate, Final.
Each completed, immutable run chunk is independently reverified
before reuse, and score rankings always use original signed Max-Cut.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import traceback

import numpy as np

from .archive import (
    _atomic_npz, _file_sha256, load_method, new_archive, new_run_id,
    save_method, update_manifest, verify_archive,
)
from .eva_cli import _git_push, _write_json
from .eva_elite_search import optimize_kl_fast
from .eva_rounding_cli import _factor_and_case, _immutable_factor_copy
from .eva_soft_em_bernoulli import soft_em, cem, gibbs_anneal
from .io import edge_arrays, load_graph
from .objective import cut_weight
from .sweep import discover_graphs


METHODS=("em-kl","em-cem","em-gibbs","em-cem-kl")
MODES=("mean","elite")
STAGES=("Source","EMHard","Candidate","Final")
_GRAPH=None
_Y=None
_EDGES=None
_IDENT=None


def _worker_init(path, source, expected):
    global _GRAPH,_Y,_EDGES,_IDENT
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    graph,y,_,graph_sha,factor_sha=_factor_and_case(
        Path(path),Path(source)
    )
    if [graph_sha,factor_sha]!=expected:
        raise ValueError("Worker source EVA factor/graph changed")
    _GRAPH,_Y,_EDGES=graph,y,edge_arrays(graph)
    _IDENT=expected


def load_elites(path, factor_sha, directory, *, limit=8):
    """Verify candidate archived experiment013 *final* partitions.

    Read score metadata for ranking, but independently verify both
    archive and original graph before admitting any elite. Never trust
    the reported 26917 without its actual bits and graph SHA.
    """
    source=Path(directory)
    if limit<1 or not source.is_dir():
        raise FileNotFoundError(
            "Elite seed directory missing; use --initializations mean "
            "or supply --elite-dir with committed Experiment013 outputs."
        )
    graph_sha=_file_sha256(path)
    source_edges=edge_arrays(load_graph(path))
    candidates=[]
    for case_path in sorted((source/"cases").glob(f"{path.stem}-grid-*.json")):
        rec=json.loads(case_path.read_text(encoding="utf8"))
        config=rec.get("settings",{})
        if (rec.get("status")!="complete"
                or config.get("graph_sha256")!=graph_sha
                or config.get("factor_sha256")!=factor_sha):
            continue
        for index,trial in enumerate(rec["trials"]):
            candidates.append((
                -float(trial["scores"]["Final"]),
                int(trial["restart"]),
                str(case_path),
                index,
                rec["archive_path"],
            ))
    if not candidates:
        raise FileNotFoundError(
            "No matching verified Experiment013 final partitions; "
            "ensure G63 experiment013 cases/archives were saved."
        )
    candidates.sort()
    elites=[]
    seen=set()
    verified_archives={}
    source_case_hashes={}
    for negscore,restart,case_path,index,archive_path in candidates:
        if len(elites)>=limit:
            break
        if archive_path not in verified_archives:
            report=verify_archive(archive_path)
            if (report["status"]!="complete"
                    or report["graph_sha256"]!=graph_sha
                    or "Grid-Final_EVA" not in report["methods"]):
                raise ValueError("Elite candidate graph/archive mismatch")
            labels,scores=load_method(
                Path(archive_path)/"Grid-Final_EVA.npz"
            )
            verified_archives[archive_path]=(labels,scores)
        labels,scores=verified_archives[archive_path]
        row=np.asarray(labels[index],dtype=np.int8)
        score=float(cut_weight(source_edges,row))
        if not (np.isclose(score,scores[index],rtol=0,atol=1e-8)
                and np.isclose(score,-negscore,rtol=0,atol=1e-8)):
            raise ValueError("Original graph elite cut score mismatch")
        # Complement normalization gives a stable canonical identity.
        standard=row if row[0]==0 else (1-row)
        canonical=np.packbits(standard,bitorder="little").tobytes()
        if canonical in seen:
            continue
        seen.add(canonical)
        if case_path not in source_case_hashes:
            source_case_hashes[case_path]=_file_sha256(case_path)
        elites.append({
            "labels":row.copy(), "score":score,
            "restart":restart,
            "source_case":case_path,
            "source_case_sha256":source_case_hashes[case_path],
            "archive":str(archive_path),
            "source_partition_sha256":_file_sha256(
                Path(archive_path)/"Grid-Final_EVA.npz"
            ),
        })
    if not elites:
        raise ValueError("No independent Experiment013 elite partitions")
    return elites


def _config(method,mode,episodes,*,graph_sha,factor_sha,budget):
    payload={
        "version":"experiment016-soft-em-bernoulli-v1",
        "method":method,"initialization":mode,
        "graph_sha256":graph_sha,
        "factor_sha256":factor_sha,
        "source_elite_hashes":[
            ep.get("source_partition_sha256") for ep in episodes
        ],
        "source_elite_scores":[
            ep.get("score") for ep in episodes
        ],
        "source_elite_restarts":[
            ep.get("restart") for ep in episodes
        ],
        "source_case_sha256":[
            ep.get("source_case_sha256") for ep in episodes
        ],
        "episodes":[int(ep["episode"]) for ep in episodes],
        "budget":budget,
    }
    key=hashlib.sha256(
        json.dumps(payload,sort_keys=True,allow_nan=False).encode()
    ).hexdigest()[:20]
    return key,payload


def _verify(record,config):
    if record.get("status")!="complete" or record.get("config")!=config:
        raise ValueError("Experiment016 cache identity mismatch")
    check=verify_archive(record["archive_path"])
    count=len(config["episodes"])
    if (check["status"]!="complete"
            or check["graph_sha256"]!=config["graph_sha256"]
            or len(record["trials"])!=count):
        raise ValueError("Experiment016 archive graph/trial mismatch")
    path=Path(record["archive_path"])
    for stage in STAGES:
        method=f"Exp016-{stage}"
        if check["methods"][method]["verified_partitions"]!=count:
            raise ValueError(f"Missing {stage} partitions")
        _,scores=load_method(path/f"{method}.npz")
        expected=[x["scores"][stage] for x in record["trials"]]
        if not np.allclose(scores,expected,rtol=0,atol=1e-8):
            raise ValueError("Experiment016 cached score/labels mismatch")
    probability_path=Path(record["probabilities_path"])
    if _file_sha256(probability_path)!=record["probabilities_sha256"]:
        raise ValueError("Saved Bernoulli probabilities changed")
    with np.load(probability_path,allow_pickle=False) as data:
        em=data["em_probabilities"]
        learned=data["search_probabilities"]
        ids=data["episode_ids"]
    if (em.shape!=(_Y.shape[0],count)
            or learned.shape!=em.shape
            or not np.isfinite(em).all()
            or not np.isfinite(learned).all()
            or np.any(em<0) or np.any(em>1)
            or np.any(learned<0) or np.any(learned>1)
            or not np.array_equal(ids,config["episodes"])):
        raise ValueError("Bernoulli probability dimensions/range mismatch")
    em_parts,_=load_method(path/"Exp016-EMHard.npz")
    for j in range(count):
        if not np.array_equal(em_parts[j],(em[:,j]>=.5).astype(np.int8)):
            raise ValueError("EM probability archive not consistent with hard cut")
    for row,idx in zip(record["trials"],config["episodes"]):
        s=row["scores"]
        if row["episode"]!=idx or (
            s["Candidate"]+1e-8<max(s["Source"],s["EMHard"])
            or s["Final"]+1e-8<s["Candidate"]
        ):
            raise ValueError("Experiment016 candidate/final monotonicity violation")
    return record


def _solve(job):
    config=job["config"]
    case=Path(job["case_path"])
    if case.is_file():
        return _verify(
            json.loads(case.read_text(encoding="utf8")),config
        ),True
    if _GRAPH is None or _Y is None or _IDENT!=[
        config["graph_sha256"],config["factor_sha256"],
    ]:
        raise RuntimeError("Worker missing original frozen EVA graph/factor")
    b=config["budget"]
    mode=config["initialization"]
    method=config["method"]
    graph_number=int(Path(job["graph_path"]).stem[1:])
    partitions={stage:[] for stage in STAGES}
    scores={stage:[] for stage in STAGES}
    em_probs=[]
    search_probs=[]
    trials=[]
    for ep in job["episodes"]:
        index=int(ep["episode"])
        rng=np.random.default_rng(np.random.SeedSequence(
            [b["seed"],graph_number,index,1601]
        ))
        anchor=ep.get("labels")
        p,emhard,emdiag=soft_em(
            _GRAPH,_Y,rng,mode=mode,anchor=anchor,
            kappa=b["kappa"],delta=b["delta"],
            temp_start=b["em_temp_start"],
            temp_end=b["em_temp_end"],
            stages=b["em_stages"],em_iters=b["em_iters"],
            damping=b["damping"],
        )
        em_score=float(cut_weight(_EDGES,emhard))
        if anchor is None:
            source=emhard.copy()
            source_score=em_score
        else:
            source=np.asarray(anchor,dtype=np.int8).copy()
            source_score=float(cut_weight(_EDGES,source))
            if not np.isclose(source_score,ep["score"],rtol=0,atol=1e-8):
                raise ValueError("Elite seed source score changed")
        candidate=source.copy() if source_score>=em_score else emhard.copy()
        candidate_score=max(source_score,em_score)
        learned=p.copy()
        details={"soft_em":emdiag}
        if method in ("em-cem","em-cem-kl"):
            # Reuse identical CEM random streams and model fitting
            # between CEM and CEM+KL for matched comparisons.
            cem_rng=np.random.default_rng(np.random.SeedSequence(
                [b["seed"],graph_number,index,1602]
            ))
            learned,ce_candidate,ce_diag=cem(
                _GRAPH,p,cem_rng,anchor=candidate,
                generations=b["cem_generations"],
                population=b["population"],
                elite_fraction=b["elite_fraction"],
                update_rate=b["update_rate"],
                min_probability=b["min_probability"],
                anchor_weight=b["anchor_weight"],
                injection=b["injection"],
                target=b["target"],
            )
            ce_cut=float(cut_weight(_EDGES,ce_candidate))
            if ce_cut>candidate_score:
                candidate=ce_candidate.copy()
                candidate_score=ce_cut
            details["cem"]=ce_diag
        if method=="em-gibbs":
            gibbs_rng=np.random.default_rng(np.random.SeedSequence(
                [b["seed"],graph_number,index,1603]
            ))
            gibbs_best,gibbs_diag=gibbs_anneal(
                _GRAPH,candidate,gibbs_rng,
                sweeps=b["gibbs_sweeps"],
                temp_start=b["gibbs_temp_start"],
                temp_end=b["gibbs_temp_end"],
                target=b["target"],
            )
            gc=float(cut_weight(_EDGES,gibbs_best))
            if gc>candidate_score:
                candidate=gibbs_best.copy()
                candidate_score=gc
            details["gibbs"]=gibbs_diag
        final=candidate.copy()
        final_score=float(candidate_score)
        if method in ("em-kl","em-cem-kl"):
            rng_kl=np.random.default_rng(np.random.SeedSequence(
                [b["seed"],graph_number,index,1604]
            ))
            kl_candidate,kl_info=optimize_kl_fast(
                _GRAPH,candidate,rng_kl,passes=b["kl_passes"],
            )
            kl_score=float(cut_weight(_EDGES,kl_candidate))
            if kl_score>final_score:
                final=kl_candidate.copy()
                final_score=kl_score
            details["kl"]=kl_info
        stage_labels={
            "Source":source,"EMHard":emhard,
            "Candidate":candidate,"Final":final,
        }
        stage_scores={
            "Source":source_score,"EMHard":em_score,
            "Candidate":float(candidate_score),
            "Final":final_score,
        }
        for stage in STAGES:
            partitions[stage].append(stage_labels[stage])
            scores[stage].append(stage_scores[stage])
        em_probs.append(p)
        search_probs.append(learned)
        trials.append({
            "episode":index,
            "method":method,"initialization":mode,
            "scores":stage_scores,
            "source_restart":ep.get("restart"),
            "diagnostics":details,
        })
    probability_path=Path(job["probability_path"])
    _atomic_npz(
        probability_path,
        em_probabilities=np.column_stack(em_probs).astype(np.float64),
        search_probabilities=np.column_stack(search_probs).astype(np.float64),
        episode_ids=np.array(config["episodes"],dtype=np.int32),
    )
    probability_sha=_file_sha256(probability_path)
    archive=new_archive(
        Path(job["output_dir"])/"archives",new_run_id(),
        Path(job["graph_path"]),_GRAPH.shape[0],_EDGES,
        2,config,jsonl_output=Path(job["trial_path"]),
        argv=["experiment016",method,mode],
    )
    for stage in STAGES:
        save_method(archive,f"Exp016-{stage}",partitions[stage],
                    scores[stage],k=2)
    update_manifest(archive,status="complete",solver_info={
        "EVA_frozen":True,
        "signed_original_graph_cut":True,
        "method":method,"initialization":mode,
        "probabilities_path":str(probability_path),
        "probabilities_sha256":probability_sha,
    })
    check=verify_archive(archive["path"])
    if (check["status"]!="complete"
            or check["graph_sha256"]!=config["graph_sha256"]
            or any(check["methods"][f"Exp016-{stage}"][
                        "verified_partitions"]!=len(trials)
                   for stage in STAGES)):
        raise RuntimeError("New Bernoulli archive did not verify")
    record={
        "status":"complete","graph":Path(job["graph_path"]).stem,
        "id":job["id"],"method":method,"initialization":mode,
        "config":config,
        "probabilities_path":str(probability_path),
        "probabilities_sha256":probability_sha,
        "archive_path":str(archive["path"]),
        "trials":trials,"best_final":float(max(scores["Final"])),
    }
    trial_path=Path(job["trial_path"])
    trial_path.parent.mkdir(parents=True,exist_ok=True)
    temp=trial_path.with_suffix(".tmp")
    with temp.open("w",encoding="utf8") as handle:
        for trial in trials:
            handle.write(json.dumps({
                "graph":record["graph"],
                "config_id":record["id"],
                **trial,
            },sort_keys=True,allow_nan=False)+"\n")
    os.replace(temp,trial_path)
    _write_json(case,record)
    return record,False


def _summaries(destination,graph_name,*,target):
    destination=Path(destination)
    path=destination/"summary.csv"
    path.parent.mkdir(parents=True,exist_ok=True)
    columns=["graph","method","initialization","episode",
             "source_cut","em_cut","candidate_cut","final_cut",
             "gain_vs_source","gap_to_target",
             "archive_path","probabilities_path","config_id"]
    temporary=path.with_suffix(".tmp")
    with temporary.open("w",encoding="utf8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        for case in sorted((destination/"cases").glob(
            f"{graph_name}-em-*.json"
        )):
            record=json.loads(case.read_text(encoding="utf8"))
            for trial in record["trials"]:
                s=trial["scores"]
                writer.writerow({
                    "graph":graph_name,"method":record["method"],
                    "initialization":record["initialization"],
                    "episode":trial["episode"],
                    "source_cut":s["Source"],"em_cut":s["EMHard"],
                    "candidate_cut":s["Candidate"],
                    "final_cut":s["Final"],
                    "gain_vs_source":s["Final"]-s["Source"],
                    "gap_to_target":target-s["Final"],
                    "archive_path":record["archive_path"],
                    "probabilities_path":record["probabilities_path"],
                    "config_id":record["id"],
                })
    os.replace(temporary,path)
    return path


def run_graph(
    path,*,factor_dir,dest,elite_dir=None,
    methods=METHODS,initializations=("mean","elite"),
    episodes=8,chunk_size=2,workers=4,seed=2026,
    kappa=1.,delta=.25,em_temp_start=5.,
    em_temp_end=.15,em_stages=8,em_iters=20,damping=.4,
    cem_generations=30,population=96,elite_fraction=.15,
    update_rate=.6,min_probability=.003,
    anchor_weight=.96,injection=.035,
    gibbs_sweeps=150,gibbs_temp_start=12.,
    gibbs_temp_end=.4,kl_passes=35,target=27047.,
):
    path, factor_dir, dest=map(Path,(path,factor_dir,dest))
    if (not methods or not initializations
            or any(x not in METHODS for x in methods)
            or any(x not in MODES for x in initializations)
            or len(set(methods))!=len(methods)
            or len(set(initializations))!=len(initializations)
            or episodes<1 or chunk_size<1 or workers<1 or seed<0
            or not np.isfinite(target)
            or min(delta,em_temp_start,em_temp_end,
                   damping,elite_fraction,update_rate,min_probability,
                   gibbs_temp_start,gibbs_temp_end)<=0
            or kappa<0 or damping>1 or elite_fraction>1
            or update_rate>1 or not 0<min_probability<.5
            or not 0<=anchor_weight<=1 or not 0<=injection<.5
            or not all(np.isfinite(v) for v in (
                kappa,delta,em_temp_start,em_temp_end,damping,
                elite_fraction,update_rate,min_probability,
                anchor_weight,injection,gibbs_temp_start,
                gibbs_temp_end,
            ))
            or em_stages<1 or em_iters<1 or cem_generations<1
            or population<2 or gibbs_sweeps<1 or kl_passes<1):
        raise ValueError("Invalid Experiment016 methods or budgets")
    graph,y,_,graph_sha,factor_sha=_factor_and_case(path,factor_dir)
    if dest.resolve()==factor_dir.resolve():
        raise ValueError("Experiment016 output cannot overwrite EVA factors")
    elites=[]
    if "elite" in initializations:
        if elite_dir is None:
            raise ValueError("--elite-dir required for elite warm starts")
        elites=load_elites(path,factor_sha,elite_dir,limit=episodes)
        print(f"{path.stem}: verified {len(elites)} unique elite starts; "
              f"best={max(x['score'] for x in elites):g}",flush=True)
    dest.mkdir(parents=True,exist_ok=True)
    _immutable_factor_copy(factor_dir,dest,path.stem,factor_sha)
    budget={
        "seed":seed,"kappa":kappa,"delta":delta,
        "em_temp_start":em_temp_start,"em_temp_end":em_temp_end,
        "em_stages":em_stages,"em_iters":em_iters,"damping":damping,
        "cem_generations":cem_generations,"population":population,
        "elite_fraction":elite_fraction,"update_rate":update_rate,
        "min_probability":min_probability,
        "anchor_weight":anchor_weight,"injection":injection,
        "gibbs_sweeps":gibbs_sweeps,
        "gibbs_temp_start":gibbs_temp_start,
        "gibbs_temp_end":gibbs_temp_end,
        "kl_passes":kl_passes,"target":target,
    }
    jobs=[]
    for mode in initializations:
        for method in methods:
            for start in range(0,episodes,chunk_size):
                subset=[]
                for index in range(start,min(start+chunk_size,episodes)):
                    row={"episode":index}
                    if mode=="elite":
                        elite=elites[index%len(elites)]
                        row.update(elite)
                    subset.append(row)
                key,config=_config(method,mode,subset,
                    graph_sha=graph_sha,factor_sha=factor_sha,budget=budget)
                name=f"{path.stem}-em-{mode}-{method}-{key}"
                jobs.append({
                    "id":key,"config":config,
                    "episodes":subset,
                    "graph_path":str(path),
                    "output_dir":str(dest),
                    "case_path":str(dest/"cases"/f"{name}.json"),
                    "probability_path":str(dest/"probabilities"/f"{name}.npz"),
                    "trial_path":str(dest/"trials"/f"{name}.jsonl"),
                })
    print(
        f"{path.stem}: original EVA Y={y.shape}; "
        f"{len(methods)} methods x {len(initializations)} init modes "
        f"x {episodes} episodes = "
        f"{len(methods)*len(initializations)*episodes} paired runs, "
        f"{len(jobs)} resumable chunks; workers={workers}; "
        f"target={target:g}",flush=True,
    )
    results=[]
    errors=[]
    best=float("-inf")
    with ProcessPoolExecutor(
        max_workers=workers,initializer=_worker_init,
        initargs=(str(path),str(factor_dir),[graph_sha,factor_sha]),
    ) as executor:
        futures={executor.submit(_solve,job):job for job in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:
                record,cached=future.result()
                results.append(record)
                best=max(best,record["best_final"])
                print(
                    f"{path.stem} {record['initialization']}/"
                    f"{record['method']} "
                    f"{'verified skip' if cached else 'complete'} "
                    f"n={len(record['trials'])} "
                    f"best={record['best_final']:g}; "
                    f"global experiment016 best={best:g} "
                    f"(gap={target-best:+g})",
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
                errors.append(issue)
                _write_json(
                    dest/"errors"/f"{Path(job['case_path']).stem}.json",
                    issue,
                )
                print(f"{path.stem} Experiment016 FAILED: {exc}",flush=True)
            _summaries(dest,path.stem,target=target)
    if errors:
        raise RuntimeError(
            f"{len(errors)} Experiment016 jobs failed; "
            f"completed cases retained. See {dest/'errors'}"
        )
    print(f"{path.stem}: Experiment016 COMPLETE "
          f"best={best:g}; gap={target-best:+g}; "
          f"summary={dest/'summary.csv'}",flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs",nargs="+",default=["G63"])
    parser.add_argument("--gset-dir",type=Path,default=Path("data"))
    parser.add_argument("--factor-dir",type=Path,default=Path(
        "results/gset/experiment003-eva-numba",
    ))
    parser.add_argument("--elite-dir",type=Path,default=Path(
        "results/gset/experiment013-eva-grid",
    ))
    parser.add_argument("--output-dir",type=Path,default=Path(
        "results/gset/experiment016-soft-em-bernoulli",
    ))
    parser.add_argument("--methods",nargs="+",choices=METHODS,
                        default=list(METHODS))
    parser.add_argument("--initializations",nargs="+",choices=MODES,
                        default=list(MODES))
    parser.add_argument("--episodes",type=int,default=8)
    parser.add_argument("--chunk-size",type=int,default=2)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--seed",type=int,default=2026)
    parser.add_argument("--kappa",type=float,default=1.)
    parser.add_argument("--delta",type=float,default=.25)
    parser.add_argument("--em-temp-start",type=float,default=5.)
    parser.add_argument("--em-temp-end",type=float,default=.15)
    parser.add_argument("--em-stages",type=int,default=8)
    parser.add_argument("--em-iters",type=int,default=20)
    parser.add_argument("--damping",type=float,default=.4)
    parser.add_argument("--cem-generations",type=int,default=30)
    parser.add_argument("--population",type=int,default=96)
    parser.add_argument("--elite-fraction",type=float,default=.15)
    parser.add_argument("--update-rate",type=float,default=.6)
    parser.add_argument("--min-probability",type=float,default=.003)
    parser.add_argument("--anchor-weight",type=float,default=.96)
    parser.add_argument("--injection",type=float,default=.035)
    parser.add_argument("--gibbs-sweeps",type=int,default=150)
    parser.add_argument("--gibbs-temp-start",type=float,default=12.)
    parser.add_argument("--gibbs-temp-end",type=float,default=.4)
    parser.add_argument("--kl-passes",type=int,default=35)
    parser.add_argument("--target",type=float,default=27047.)
    parser.add_argument("--push",action="store_true")
    parser.add_argument("--push-branch")
    args=parser.parse_args()
    if args.push and not args.push_branch:
        parser.error("--push requires --push-branch")
    if args.output_dir.resolve()==args.factor_dir.resolve():
        parser.error("Output cannot overwrite source EVA factor directory")
    failures=0
    paths=discover_graphs(args.gset_dir,args.graphs)
    for path in paths:
        try:
            run_graph(
                path,factor_dir=args.factor_dir,dest=args.output_dir,
                elite_dir=args.elite_dir,
                methods=tuple(args.methods),
                initializations=tuple(args.initializations),
                episodes=args.episodes,
                chunk_size=args.chunk_size,workers=args.workers,
                seed=args.seed,kappa=args.kappa,delta=args.delta,
                em_temp_start=args.em_temp_start,
                em_temp_end=args.em_temp_end,
                em_stages=args.em_stages,em_iters=args.em_iters,
                damping=args.damping,
                cem_generations=args.cem_generations,
                population=args.population,
                elite_fraction=args.elite_fraction,
                update_rate=args.update_rate,
                min_probability=args.min_probability,
                anchor_weight=args.anchor_weight,
                injection=args.injection,
                gibbs_sweeps=args.gibbs_sweeps,
                gibbs_temp_start=args.gibbs_temp_start,
                gibbs_temp_end=args.gibbs_temp_end,
                kl_passes=args.kl_passes,target=args.target,
            )
        except (OSError,ValueError,RuntimeError,AssertionError,
                KeyError,ArithmeticError) as exc:
            failures+=1
            _write_json(args.output_dir/"errors"/f"{path.stem}.json",{
                "graph":path.stem,
                "error":f"{type(exc).__name__}: {exc}",
                "traceback":traceback.format_exc().replace(
                    str(Path.home()),"<HOME>",
                ),
            })
            print(f"{path.stem}: Experiment016 ERROR {exc}",flush=True)
            break
    if args.push and failures==0:
        _git_push(
            args.output_dir,"experiment016 soft EM CEM Gibbs Bernoulli",
            allowed_branch=args.push_branch,
        )
    print(
        f"Experiment016: graphs={len(paths)}, errors={failures}; "
        f"summary={args.output_dir/'summary.csv'}",flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__=="__main__":
    main()
