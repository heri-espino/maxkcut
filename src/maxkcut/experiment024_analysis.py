"""Topology-aware descriptive summary for complete 16-GW-seed Experiment024 rows.

Reports graph-level (not restart-level) method differences to GW-RR, shared
wins, fractional ties, normalized gaps to archived feasible values and time.
Incomplete graph-method cells are preserved but never counted as winners.
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

from .experiment024_paired_gw16 import BASE, METHODS, GRAPH_IDS, coverage, atomic_json
from .experiment023_references import load_references
from .graph_structure_methods import labels as graph_labels

def summarize(root=BASE,phase="full"):
    out=Path(root)/"analysis"
    # Always update the 1278-cell manifest; absent methods remain explicit.
    manifest=coverage(root=root,phase=phase)
    path=out/(phase+"_coverage.csv")
    with path.open(encoding="utf8",newline="") as stream:
        records=list(csv.DictReader(stream))
    lookup={(r["graph"],r["method"]):r for r in records}
    refs,_=load_references()
    by_graph=[]
    for graph in GRAPH_IDS:
        family,weight_type=graph_labels(graph)
        gw=lookup[(graph,"gw-rr")]
        baseline=float(gw["refined_best"]) if gw["status"]=="complete_16" else None
        eligible=[r for r in records if r["graph"]==graph and r["status"]=="complete_16"]
        top=max([float(r["refined_best"]) for r in eligible],default=None)
        ties=[r for r in eligible if top is not None and float(r["refined_best"])==top]
        complete_graph=len(eligible)==len(METHODS)
        for method in METHODS:
            row=lookup[(graph,method)]
            valid=row["status"]=="complete_16"
            value=float(row["refined_best"]) if valid else None
            reference=float(refs[graph]["reference_cut"])
            by_graph.append(dict(graph=graph,family=family,weight_class=weight_type,
                method=method,status=row["status"],graph_methods_complete=int(complete_graph),
                raw_best=float(row["raw_best"]) if valid else "",
                raw_mean_from_16=(float(row["raw_mean"]) if valid and row["raw_mean"] not in ("", "None") else ""),
                best_refined_cut=value if valid else "",
                reference_cut=reference,
                reference_gap=reference-value if valid else "",
                fraction_reference=value/reference if valid and reference>0 else "",
                gw_rr_cut=baseline if baseline is not None else "",
                delta_to_gw=value-baseline if valid and baseline is not None else "",
                relative_delta_to_gw=(value-baseline)/max(abs(baseline),1)
                    if valid and baseline is not None else "",
                shared_best=int(value==top) if valid else "",
                fractional_best=1/len(ties) if valid and value==top else 0 if valid else "",
                wall_seconds=row["wall_seconds"] if valid else "",
                compare_mst_hybrid=int(method.startswith("mst-"))))
    table=out/(phase+"_method_by_graph.csv")
    with table.open("w",newline="",encoding="utf8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(by_graph[0]))
        writer.writeheader()
        writer.writerows(by_graph)
    grouped=defaultdict(list)
    for r in by_graph:
        grouped[(r["family"],r["method"])].append(r)
    summary=[]
    for family in ("random","planar","toroidal"):
        for method in METHODS:
            rows=grouped[family,method]
            complete=[r for r in rows if r["status"]=="complete_16"]
            paired=[r for r in complete if r["delta_to_gw"]!=""]
            wins=[r for r in complete if r["shared_best"]!=""]
            summary.append(dict(family=family,method=method,graphs=len(rows),
                complete_16=len(complete),matched_gw_pairs=len(paired),
                shared_best_wins=sum(int(r["shared_best"]) for r in wins),
                fractional_wins=sum(float(r["fractional_best"]) for r in wins),
                mean_relative_gain_vs_gw=(sum(float(r["relative_delta_to_gw"]) for r in paired)/len(paired)) if paired else "",
                mean_gap_to_reference=(sum(float(r["reference_gap"]) for r in complete)/len(complete)) if complete else "",
                mean_wall_seconds=(sum(float(r["wall_seconds"]) for r in complete)/len(complete)) if complete else "",
                warning="MST+GW hybrid; deterministic tree" if method.startswith("mst-") else "method-internal proposal budgets vary"))
    with (out/(phase+"_method_by_family.csv")).open("w",newline="",encoding="utf8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    report=dict(phase=phase,complete_16_method_graph_cells=manifest["complete_16_cells"],
         total_cells=manifest["planned_cells"],entire_study_complete=manifest["primary_analysis_valid"],
         paired_baseline="gw-rr; identical 16 normals for all methods",
         methodology="best raw of 16 matched starts, one common full refinement",
         mst="GW-anchored deterministic MST hybrids, not 16 independent MST randomizations",
         warning="Descriptive only: method-internal cut-evaluation and wall-time budgets are not equal.",
         files=[str(table),str(out/(phase+"_method_by_family.csv"))])
    atomic_json(out/(phase+"_topology_report.json"),report)
    return report

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--phase",default="full",choices=["pilot","full"])
    a=p.parse_args()
    print(json.dumps(summarize(phase=a.phase),indent=2))
