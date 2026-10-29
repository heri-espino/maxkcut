"""Experiment020: audit all available saved G-set EVA factors and compare
nine KMeans initializers with 50 matched full Flip→KL→ILS restarts.

The inventory discovers only *completed, hash-verifiable* original
EVA factors: Experiment002 (historically G1..G59) and Experiment003
Numba (G63 G64 G65 G66 G67 G70 G72 G77 G81). Incomplete factors
(e.g. G60) and graphs without factors are reported, NEVER rebuilt.

G63's already-completed 450-run Experiment019 is reused unchanged.
Other graphs are computed in isolated per-graph folders in Exp020
using all 32 requested independent Windows worker processes.
Separate original signed graph identities and factor snapshots
are mandatory: no numerical score is compared across graphs.

Produces 30k+ long-format rows and within-instance paired deltas
with exact restart alignment. Use bootstrap CIs only across restarts
of the *same* graph, not pooled raw scores across different graphs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .eva_cli import _write_json
from .eva_rounding_cli import _factor_and_case
from .eva_kmeans_initializers import INIT2

BASE=Path("results/gset")
SOURCES=(
    BASE/"experiment003-eva-numba",
    BASE/"experiment002-eva",
)
REUSED_G63=BASE/"experiment019-kmeans-init-full-pipeline"
RESULT_ROOT=BASE/"experiment020-all-gset-paired-inits"
COLUMNS=("graph","source","initializer","restart","rounding_seed",
         "search_seed","rounding_seed_stream","raw_cut",
         "flip_cut","kl_cut","final_cut","improvement",
         "seconds","extra_start_cuts","archive_path","case_id")


def inventory(*,data_dir=Path("data"),out=RESULT_ROOT,
              sources=SOURCES,reused=REUSED_G63):
    data_dir=Path(data_dir)
    out=Path(out)
    sources=tuple(map(Path,sources))
    graph_paths=sorted(data_dir.glob("G[0-9]*.txt"),
        key=lambda x:int(x.stem[1:]))
    rows=[]
    for path in graph_paths:
        n=int(path.stem[1:])
        record={"graph":path.stem,"vertices":None,
                "graph_path":str(path),
                "status":"no_completed_EVA_factor",
                "factor_source":None,"factor_sha256":None,
                "graph_sha256":None,
                "output_dir":str(out/path.stem)}
        for source in sources:
            case=source/"cases"/f"{path.stem}.json"
            factor=source/"factors"/f"{path.stem}.npz"
            if not (case.is_file() and factor.is_file()):
                continue
            try:
                graph,y,metadata,g_sha,f_sha=_factor_and_case(
                    path,source,
                )
            except (ValueError,KeyError,OSError,TypeError,
                    AttributeError) as exc:
                # Do not hide invalid factors; include the graph
                # and exact source and error in coverage report.
                record.update({
                    "status":"invalid_existing_EVA",
                    "factor_source":str(source),
                    "error":f"{type(exc).__name__}: {exc}",
                })
                continue
            record.update({
                "vertices":int(graph.shape[0]),
                "rank":int(y.shape[1]),
                "factor_source":str(source),
                "factor_sha256":f_sha,
                "graph_sha256":g_sha,
                "status":"ready",
            })
            if path.stem=="G63":
                record["output_dir"]=str(reused)
                record["status"]="reused_019"
            break
        rows.append(record)
    if not rows:
        raise ValueError("No graph files found")
    out.mkdir(parents=True,exist_ok=True)
    ready=sum(x["status"]=="ready" for x in rows)
    reused_count=sum(x["status"]=="reused_019" for x in rows)
    ignored=[x for x in rows
             if x["status"] not in ("ready","reused_019")]
    payload={
        "version":"experiment020-inventory-v1",
        "factor_sources":[str(s) for s in sources],
        "saved_G63_source":str(reused),
        "graph_count":len(rows),
        "verified_ready":ready,
        "reused_verified_candidate":reused_count,
        "unavailable_count":len(ignored),
        "unavailable":[
            {"graph":x["graph"],"status":x["status"],
             **({"error":x["error"]} if "error" in x else {})}
            for x in ignored
        ],
        "records":rows,
    }
    _write_json(out/"inventory.json",payload)
    print(
        f"Experiment020 inventory: total={len(rows)}, "
        f"valid new graph factors={ready}, "
        f"reuse G63={reused_count}, "
        f"unavailable={len(ignored)}",
        flush=True,
    )
    if ignored:
        print("Unavailable:",", ".join(
            f"{r['graph']} ({r['status']})" for r in ignored
        ),flush=True)
    return payload


def _read_long_rows(record,*,restarts):
    path=Path(record["output_dir"])/"summary.csv"
    if not path.is_file():
        return None
    with path.open("r",encoding="utf8",newline="") as inp:
        rows=list(csv.DictReader(inp))
    expected={(init,index) for init in INIT2
              for index in range(restarts)}
    actual={(r["initializer"],int(r["restart"])) for r in rows}
    if len(rows)!=len(expected) or actual!=expected:
        # An incomplete graph should never be called statistically
        # complete or committed as if its 9x50 runs had finished.
        return None
    source=record["factor_source"]
    b=[]
    for r in rows:
        f={k:float(r[k]) for k in (
            "raw_cut","flip_cut","kl_cut","final_cut",
            "seconds",
        )}
        if not (
            f["raw_cut"]<=f["flip_cut"]+1e-8<=f["kl_cut"]+2e-8
            and f["kl_cut"]<=f["final_cut"]+1e-8
        ):
            raise ValueError(
                f"Nonmonotone archived cut {record['graph']} "
                f"{r['initializer']} restart {r['restart']}"
            )
        stream=0 if r["initializer"]=="paired" else 1
        b.append({
            "graph":record["graph"],
            "source":source,
            "initializer":r["initializer"],
            "restart":int(r["restart"]),
            "rounding_seed":2026,
            "search_seed":2026,
            "rounding_seed_stream":stream,
            **f,
            "improvement":f["final_cut"]-f["raw_cut"],
            "extra_start_cuts":int(r["extra_start_cuts"]),
            "archive_path":r["archive_path"],
            "case_id":r["case_id"],
        })
    return b


def _bootstrap_mean_ci(values,seed=2026,samples=4000):
    x=np.asarray(values,dtype=np.float64)
    if len(x)<2:
        return float(x.mean()),float(x.mean())
    rng=np.random.default_rng(seed)
    # Percentile bootstrap over PAIRED restart differences,
    # reproducible from seed and number of bootstraps.
    draws=rng.integers(0,len(x),size=(samples,len(x)))
    estimates=np.mean(x[draws],axis=1)
    return [float(v) for v in np.quantile(estimates,[.025,.975])]


def aggregate(*,out=RESULT_ROOT,restarts=50,
              bootstrap_seed=2026,bootstrap_samples=4000,
              require_complete=False):
    out=Path(out)
    manifest_path=out/"inventory.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest=json.loads(manifest_path.read_text(encoding="utf8"))
    all_rows=[]
    coverage=[]
    for record in manifest["records"]:
        status=record["status"]
        if status in ("ready","reused_019"):
            rows=_read_long_rows(record,restarts=restarts)
            if rows is None:
                status="missing_or_incomplete_experiment"
            else:
                status=("reused_verified_candidate" if
                        record["status"]=="reused_019" else "complete")
                all_rows.extend(rows)
        coverage.append({
            "graph":record["graph"],
            "status":status,
            "source":record["factor_source"],
            "n_restarts":(restarts if status in (
                "complete","reused_verified_candidate"
            ) else 0),
            "expected_methods":len(INIT2),
            "full_output":record["output_dir"],
        })
    if require_complete and any(
        x["status"] not in (
            "complete","reused_verified_candidate",
            "no_completed_EVA_factor",
        ) for x in coverage
    ):
        raise RuntimeError(
            "One or more available saved factors lacks complete "
            "9x50 runs. Refuse to report partial comparisons as final."
        )
    statistics=out/"statistics"
    statistics.mkdir(parents=True,exist_ok=True)
    rows_sorted=sorted(all_rows,key=lambda r:(
        int(r["graph"][1:]),r["initializer"],r["restart"]
    ))
    long_path=statistics/"all_runs.csv"
    temp=long_path.with_suffix(".tmp")
    with temp.open("w",encoding="utf8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows_sorted)
    temp.replace(long_path)
    by_graph={}
    for row in all_rows:
        by_graph.setdefault(row["graph"],{}).setdefault(
            row["initializer"],{}
        )[row["restart"]]=row
    summaries=[]
    paired=[]
    for graph in sorted(by_graph,key=lambda x:int(x[1:])):
        groups=by_graph[graph]
        reference=groups["paired"]
        for init in INIT2:
            group=groups[init]
            ids=sorted(set(group)&set(reference))
            if len(ids)!=restarts:
                raise RuntimeError(
                    f"Missing matched restarts {graph}/{init}"
                )
            final=np.asarray([
                group[i]["final_cut"] for i in ids
            ])
            base=np.asarray([
                reference[i]["final_cut"] for i in ids
            ])
            diff=final-base
            times=np.asarray([
                group[i]["seconds"] for i in ids
            ])
            delta_ci=_bootstrap_mean_ci(
                diff,seed=bootstrap_seed+
                     int(graph[1:])*17+INIT2.index(init),
                samples=bootstrap_samples,
            )
            summaries.append({
                "graph":graph,"initializer":init,
                "restarts":restarts,
                "best_final":float(final.max()),
                "mean_final":float(final.mean()),
                "median_final":float(np.median(final)),
                "std_final":float(final.std(ddof=1)),
                "mean_seconds":float(times.mean()),
                "best_restart":int(ids[int(np.argmax(final))]),
                "additional_initial_cut_evaluations":int(
                    group[ids[0]]["extra_start_cuts"]
                ),
            })
            paired.append({
                "graph":graph,
                "initializer":init,
                "control":"paired",
                "n_matched":len(ids),
                "mean_delta":float(diff.mean()),
                "median_delta":float(np.median(diff)),
                "sd_delta":float(diff.std(ddof=1)),
                "best_delta":float(diff.max()),
                "wins":int(np.count_nonzero(diff>0)),
                "ties":int(np.count_nonzero(diff==0)),
                "losses":int(np.count_nonzero(diff<0)),
                "bootstrap_mean_delta_ci_low":delta_ci[0],
                "bootstrap_mean_delta_ci_high":delta_ci[1],
                "bootstrap_seed":bootstrap_seed+
                        int(graph[1:])*17+INIT2.index(init),
                "bootstrap_draws":bootstrap_samples,
            })
    for name,data in (
        ("per_instance_methods.csv",summaries),
        ("paired_vs_original_kmeansnm.csv",paired),
        ("coverage.csv",coverage),
    ):
        fields=list(data[0]) if data else ["graph","status"]
        destination=statistics/name
        temp=destination.with_suffix(".tmp")
        with temp.open("w",encoding="utf8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
        temp.replace(destination)
    report={
        "version":"experiment020-statistics-v1",
        "graph_count":manifest["graph_count"],
        "complete_graphs":len(by_graph),
        "completed_new_graphs":sum(
            r["status"]=="complete" for r in coverage
        ),
        "reused_existing_graphs":sum(
            r["status"]=="reused_verified_candidate"
            for r in coverage
        ),
        "no_EVA_factor_count":sum(
            r["status"]=="no_completed_EVA_factor"
            for r in coverage
        ),
        "incomplete_candidate_count":sum(
            r["status"]=="missing_or_incomplete_experiment"
            for r in coverage
        ),
        "runs_in_all_runs_csv":len(all_rows),
        "methods":list(INIT2),
        "matched_restarts_per_graph":restarts,
        "seed":2026,
        "paired_bootstrap_draws":bootstrap_samples,
        "paired_bootstrap_seed":bootstrap_seed,
        "notes":[
            "Original signed Max-Cut scores; only compare within graph.",
            "Every finalized graph has each method on the same restart IDs.",
            "Each row contains actual stored score & archive path; "
            "archive independent verification occurs during each source run.",
            "G63 is reused from successful 450-run Experiment019.",
            "Bootstrap intervals are across fixed restart IDs, "
            "not confidence intervals for global Max-Cut optimality.",
            "Random seed schedule: KMeans SeedSequence([2026,graphNumber,2, "
            "restart,0 if paired else 1]); "
            "Flip [2026,graphNumber,restart,1313], "
            "KL [2026,graphNumber,restart,101], "
            "ILS [2026,graphNumber,restart,1401].",
        ],
    }
    _write_json(statistics/"report.json",report)
    print("Experiment020 aggregated:",
          f"{len(by_graph)} complete graphs,",
          f"{len(all_rows)} signed partitions/runs,",
          f"{report['incomplete_candidate_count']} incomplete.",
          flush=True)
    if require_complete and report["incomplete_candidate_count"]:
        raise RuntimeError("Incomplete factors detected in final aggregation")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("inventory","aggregate"))
    parser.add_argument("--data-dir",type=Path,default=Path("data"))
    parser.add_argument("--output-root",type=Path,default=RESULT_ROOT)
    parser.add_argument("--restarts",type=int,default=50)
    parser.add_argument("--require-complete",action="store_true")
    args=parser.parse_args()
    if args.phase=="inventory":
        inventory(data_dir=args.data_dir,out=args.output_root)
    else:
        aggregate(out=args.output_root,restarts=args.restarts,
                  require_complete=args.require_complete)


if __name__=="__main__":
    main()
