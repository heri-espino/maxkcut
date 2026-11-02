"""Experiment022 extension: retrospective, provenance-aware historical baselines.

This does NOT run EVA, SDP, MST, EM, KL, ILS, or any optimization.
It aggregates existing published result CSVs into two NON-equivalent layers:
(1) three initial rounding algorithms on 62 graphs, with 20 starts each;
(2) archived historical methods on G63 alone, with their original budgets.

Run from repository root with: python -m maxkcut.historical_structure_comparison
Requires the immutable results/gset/experiment{002,003,008,009,012,013,
016,017,018,019} folders. Fails closed on absent or duplicate input.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .graph_structure_methods import labels

RESULTS = Path("results/gset")
DEST = RESULTS / "experiment022-graph-structure-methods"

SOURCES = {
    "002": "experiment002-eva/summary.csv",
    "003": "experiment003-eva-numba/summary.csv",
    "008": "experiment008-eva-rounding-flips/summary.csv",
    "009": "experiment009-mst-multiedge/summary.csv",
    "012": "experiment012-cut-aware-kmeans-kl/summary.csv",
    "013": "experiment013-eva-grid/summary.csv",
    "016": "experiment016-soft-em-bernoulli/summary.csv",
    "017": "experiment017-multigroup-coarsening/summary.csv",
    "018": "experiment018-kmeans-initializations/summary.csv",
    "019": "experiment019-kmeans-init-full-pipeline/summary.csv",
}
# Declared historical result families with no verified consolidated summary at
# their canonical documented paths on the audited source commit. This is NOT
# evidence that no underlying logs or archived partitions exist elsewhere.
MISSING_SUMMARY_FAMILIES = {
    "005": "Single-edge MST: no original 005 summary; 009 q=1 retains a derived single-edge baseline",
    "006": "Central-edge MST: no original 006 summary found",
    "010": "Generalized KL/Tabu: no original 010 consolidated summary found",
    "011": "Tabu/elite/ILS: no original 011 consolidated summary found",
    "015": "Direct hyperplane optimization: no original 015 consolidated summary found",
}
RAW = ("GW/FJ_EVA", "KMeans2N_EVA", "KMeansNM_EVA")
RAW_FIELDS = ("eva_FJ_best", "eva_KMeans2N_best", "eva_KMeansNM_best")


def read_csv(path: Path) -> list[dict[str, str]]:
    """Require complete original summary; silently missing data are forbidden."""
    with path.open(newline="", encoding="utf8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError(f"Empty CSV: {path}")
        return list(r)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def must_g63(rows: list[dict[str, str]], source: str):
    if not rows or {r["graph"] for r in rows} != {"G63"}:
        raise ValueError(f"{source}: expected G63-only saved data")


def finite_cut(value: str, source: str) -> float:
    try:
        cut = float(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{source}: missing or invalid cut") from e
    if not (-1e100 < cut < 1e100):
        raise ValueError(f"{source}: nonfinite cut")
    return cut


def _rows_by(rows, fields):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[x] for x in fields)].append(r)
    return dict(sorted(out.items()))


def build_raw(rows002, rows003, rows008):
    """20-restart best scores for GW/FJ, independent 2-means, antipodal 2-means.

    G1-G59 from 002; G77/G81 from 003; G63 from the *unflipped* 008
    methods. No inference about the twelve other 002 placeholder rows.
    """
    old = {}
    for source, rows in (("002", rows002), ("003", rows003)):
        for r in rows:
            if r["status"] != "converged":
                continue
            if r["graph"] in old:
                raise ValueError(f"duplicate graph in old EVA summaries: {r['graph']}")
            old[r["graph"]] = (source, dict(zip(RAW, [finite_cut(r[f],source) for f in RAW_FIELDS])))
    if {g for g in old} != {f"G{i}" for i in range(1,60)} | {"G77", "G81"}:
        raise ValueError("Expected 59 original EVA and 2 Numba rounding baselines")
    must_g63(rows008, "008")
    by_method = {r["method"]: r for r in rows008}
    if len(by_method) != len(rows008):
        raise ValueError("duplicate 008 method")
    selection = ["FJ_EVA", "K-Means2N_EVA", "K-MeansNM_EVA"]
    if any(m not in by_method or int(by_method[m]["rounding_count"]) != 20 for m in selection):
        raise ValueError("008 G63 missing one of three 20-start raw rounders")
    old["G63"] = ("008", dict(zip(RAW, [finite_cut(by_method[m]["best_cut"],m) for m in selection])))
    result = []
    for graph in sorted(old, key=lambda k:int(k[1:])):
        exp, method_scores = old[graph]
        family, sign = labels(graph)
        best=max(method_scores.values())
        winners = [m for m in RAW if method_scores[m]==best]
        base = method_scores["GW/FJ_EVA"]
        for m in RAW:
            cut = method_scores[m]
            result.append(dict(graph=graph,family=family,weight_type=sign,source_experiment=exp,
                method=m,starts=20,best_raw_cut=cut,gain_vs_GW=cut-base,
                relative_gain_vs_GW=(cut-base)/max(abs(base),1.),
                shared_best=int(cut==best),fractional_best=1/len(winners) if cut==best else 0.,
                stage="initial_rounding_only",equal_restart_count=1, comparable_to_exp021_final=0))
    return result


def aggregate_raw(rows):
    out=[]
    for fam in ("random","planar","toroidal"):
        for method in RAW:
            group=[r for r in rows if r["family"]==fam and r["method"]==method]
            if not group:
                continue
            n=len(group)
            out.append(dict(family=fam,method=method,graphs=n,
                shared_best=sum(r["shared_best"] for r in group),
                fractional_best=sum(r["fractional_best"] for r in group),
                mean_raw_gain_vs_GW=sum(r["gain_vs_GW"] for r in group)/n,
                mean_relative_gain_percent=100*sum(r["relative_gain_vs_GW"] for r in group)/n,
                pipeline="EVA factor -> 20 raw roundings only; no Flip/KL/ILS"))
    return out


def build_g63(datasets):
    """One row per HISTORICAL method/configuration family on G63 only.

    A per-family *best* is a maximum across differing configurations;
    `n_records` must never be interpreted as exchangeable restarts.
    """
    records=[]
    def add(exp,name,subset,cutkey,*,warm=False,notes="",configs=(),sourcekey=None,rawkey=None,stage="not_same_as_021_full_pipeline"):
        if not subset:
            raise ValueError(f"{exp}: {name} has no observed records")
        best=max(finite_cut(r[cutkey],f"{exp}:{name}") for r in subset)
        src=max(finite_cut(r[sourcekey],f"{exp}:{name}:source") for r in subset) if sourcekey else None
        raw=max(finite_cut(r[rawkey],f"{exp}:{name}:raw") for r in subset) if rawkey else None
        gains=[finite_cut(r[cutkey],name)-finite_cut(r[sourcekey],name) for r in subset] if sourcekey else []
        records.append(dict(graph="G63",family="planar",experiment=exp,method=name,
            records=len(subset),configurations=(len({tuple(r[x] for x in configs) for r in subset}) if configs else ""),
            best_cut=best,max_source_cut=(src if src is not None else ""),max_raw_cut=(raw if raw is not None else ""),
            max_per_record_gain_from_source=(max(gains) if gains else ""),
            positive_improvements_on_own_source=(sum(g>0 for g in gains) if gains else ""),
            retained_elite_source=int(warm),stage=stage,notes=notes,
            source_csv=SOURCES[exp]))
    r=datasets["008"]
    for method,a in _rows_by(r,("method",)).items():
        add("008",method[0],a,"best_cut",notes="20 raw starts or 1000 positive-flip starts; different search budgets")
    r=datasets["009"]
    for (metric,),a in _rows_by(r,("metric",)).items():
        add("009",f"MST-multi-edge-{metric}",a,"final_cut",rawkey="best_coarse_cut",
            configs=("q",),notes="Eight q settings per distance; graph vertex-flip stage; 1 deterministic MST per metric")
        q1=[v for v in a if int(v["q"])==1]
        add("009",f"MST-single-edge-{metric} (derived from 009)",q1,"longest_edge_baseline",
            notes="Source 005 direct summary unavailable; only the 009 q=1 archived single-edge score")
    r=datasets["012"]
    for (stage,),a in _rows_by(r,("stage",)).items():
        add("012",f"cut-aware/{stage}",a,"cut",notes="200 source starts; KL variants only for stage names ending KL")
    r=datasets["013"]
    for (family,),a in _rows_by(r,("family",)).items():
        add("013",family[0],a,"final_cut",sourcekey="source_cut",rawkey="round_cut",
            configs=("config_id",),notes="Best over multiple full-grid configurations; possible reuse of prior elite/source starts")
    r=datasets["016"]
    for (init,method),a in _rows_by(r,("initialization","method")).items():
        add("016",f"EM-{init}-{method}",a,"final_cut",sourcekey="source_cut",rawkey="em_cut",warm=init=="elite",
            configs=("config_id",),notes="Four episodes; elite source retention can create a tie without finding a new cut")
    r=datasets["017"]
    for (k,init),a in _rows_by(r,("k","init")).items():
        add("017",f"multigroup-k{k}-{init}",a,"post_KL_cut",rawkey="raw_cut",configs=("k","init"),
            notes="Two-start pilot; k-way labels merged EXACTLY to binary, then KL")
    r=datasets["018"]
    for (init,),a in _rows_by(r,("init",)).items():
        add("018",f"kmeans-init-pilot-{init}",a,"post_KL_cut",rawkey="raw_cut",
            notes="Three-start KL-only pilot; not the 50-restart full-pipeline Experiment019")
    r=datasets["019"]
    for (init,),a in _rows_by(r,("initializer",)).items():
        add("019",f"full-pipeline-{init}",a,"final_cut",rawkey="raw_cut",
            stage="Round-Flip-KL-ILS",notes="50 matched starts; identical pipeline to Experiment021")
    return sorted(records,key=lambda x:(int(x["experiment"]),x["method"]))


def write_csv(path,rows):
    if not rows:
        raise ValueError(f"No records to write: {path}")
    with Path(path).open("w",encoding="utf8",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)


def execute(results_dir=RESULTS, out=DEST):
    results_dir,out=Path(results_dir),Path(out)
    data={k:read_csv(results_dir/p) for k,p in SOURCES.items()}
    for k,rows in data.items():
        if k not in {"002","003"}:
            must_g63(rows,k)
    rounders=build_raw(data["002"],data["003"],data["008"])
    families=aggregate_raw(rounders)
    specialized=build_g63(data)
    if len(rounders)!=62*3:
        raise ValueError("Incomplete 62-graph three-rounder dataset")
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"historical_raw_rounding_by_graph.csv",rounders)
    write_csv(out/"historical_raw_rounding_by_family.csv",families)
    write_csv(out/"historical_g63_methods.csv",specialized)
    coverage=[]
    for exp,path in SOURCES.items():
        observations=rounders if exp in {"002","003"} else data[exp]
        seen={r["graph"] for r in observations if r.get("graph")}
        if exp in {"002","003"}:seen={r["graph"] for r in data[exp] if r.get("status")=="converged"}
        coverage.append(dict(experiment=exp,source=path,source_status="verified_csv",graphs=len(seen),graph_ids=";".join(sorted(seen,key=lambda g:int(g[1:]))),rows=len(data[exp]),
            score_stage="raw_rounding_only" if exp in {"002","003"} else "mixed_historical_protocol",caveat="not equal budget or refinement across experiments"))
    for exp,why in MISSING_SUMMARY_FAMILIES.items():
        coverage.append(dict(experiment=exp,source="",source_status="no_verified_consolidated_summary",graphs=0,graph_ids="",rows=0,score_stage="unknown",caveat=why))
    coverage.sort(key=lambda r:int(r["experiment"]))
    write_csv(out/"historical_method_coverage.csv",coverage)
    byfam={x["family"]:x["graphs"] for x in families if x["method"]==RAW[0]}
    report=dict(status="retrospective_not_equal_budget",total_graphs_from_raw_rounders=62,
        raw_rounding_graph_counts=byfam,total_raw_rounding_rows=len(rounders),
        g63_historical_method_variants=len(specialized),raw_rounding_restarts_per_method_graph=20,
        source_sha256={k:sha(results_dir/p) for k,p in SOURCES.items()},
        missing_historical_summaries=MISSING_SUMMARY_FAMILIES,
        note="EM and MST are only G63 in these sources; no across-topology family inference possible",
        elite_warning="EM elite starts copy 26917 feasible source; 26917 tie does not constitute improvement",
        comparability="G63 historical variants use heterogeneous stage, search budget, config selection and warm starts. Do not pool with Experiment021 50-start results.")
    (out/"historical_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf8")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir",type=Path,default=RESULTS)
    p.add_argument("--out",type=Path,default=DEST)
    a=p.parse_args()
    print(json.dumps(execute(a.results_dir,a.out),indent=2))


if __name__=="__main__":main()