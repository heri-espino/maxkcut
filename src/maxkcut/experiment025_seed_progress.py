"""Cumulative best-cut trajectories after successive matched GW seeds.

This is a descriptive status report, NOT an independent partition validator.
The original per-seed saver verifies the signed score/partition before
publishing it. A final full-archive audit must still verify every NPZ.
A graph/method is eligible at seed k only with ALL seeds 1..k present.
Rank/tie counts require ALL 18 methods for the same graph and seed prefix.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

from .experiment025_bestcut import ROOT
from .experiment024_paired_gw16 import GRAPH_IDS, METHODS
from .graph_structure_methods import labels as family_of_graph
from .experiment023_references import load_references
from .experiment023_methods import atomic_json


def cumulative(*,root=ROOT,phase="full"):
    if phase not in ("pilot","full"):
        raise ValueError(phase)
    graphs=("G1","G14","G11") if phase=="pilot" else GRAPH_IDS
    refs,_=load_references()
    per_case={}
    audit=[]
    for graph in graphs:
        family,wt=family_of_graph(graph)
        for method in METHODS:
            base=Path(root)/phase/graph/method
            completed={}
            for s in range(16):
                file=base/f"{s:04d}.json"
                if not file.exists():
                    continue
                d=json.loads(file.read_text(encoding="utf8"))
                if (d.get("graph")!=graph or d.get("method")!=method
                        or int(d.get("index",-1))!=s
                        or "best_seen_cut" not in d):
                    raise ValueError(f"Invalid per-seed metadata: {file}")
                score=float(d["best_seen_cut"])
                if abs(score-float(d["stages"]["Final"]))>1e-8:
                    raise ValueError(f"Saved highest signed cut != stage Final: {file}")
                if not (base/f"{s:04d}.npz").is_file():
                    raise ValueError(f"Missing binary partition archive {file}")
                completed[s]=score
            per_case[(graph,method)]=completed
            audit.append(dict(graph=graph,family=family,weight_class=wt,
                method=method,completed_seed_indices=";".join(str(x+1) for x in sorted(completed)),
                count_completed=len(completed),best_any_seed=max(completed.values()) if completed else ""))
    rows=[]
    for k in range(1,17):
        for graph in graphs:
            per_method={}
            for method in METHODS:
                saved=per_case[(graph,method)]
                if all(s in saved for s in range(k)):
                    per_method[method]=max(saved[s] for s in range(k))
            all_methods=len(per_method)==len(METHODS)
            top=max(per_method.values()) if all_methods else None
            gw=per_method.get("gw-rr")
            family,wt=family_of_graph(graph)
            for method in METHODS:
                cut=per_method.get(method)
                eligible=cut is not None
                rows.append(dict(seed_prefix=k,graph=graph,family=family,
                    weight_class=wt,method=method,
                    eligible=int(eligible),all_18_methods_matched=int(all_methods),
                    best_seen_cut=cut if eligible else "",
                    paired_gw_best=gw if gw is not None else "",
                    gain_vs_gw=cut-gw if eligible and gw is not None else "",
                    tied_graph_best=int(cut==top) if all_methods else "",
                    archived_reference_cut=refs[graph]["reference_cut"]))
    out=Path(root)/"analysis"
    out.mkdir(parents=True,exist_ok=True)
    for path,data in ((out/(phase+"_seed_progress.csv"),rows),
                      (out/(phase+"_seed_completion.csv"),audit)):
        with path.open("w",newline="",encoding="utf8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    graph_pairs=len(graphs)*len(METHODS)
    prefixes={}
    for k in range(1,17):
        slice_k=[r for r in rows if r["seed_prefix"]==k]
        prefixes[str(k)]=dict(
            completed_graph_method_prefixes=sum(int(r["eligible"]) for r in slice_k),
            total_graph_method_pairs=graph_pairs,
            fully_paired_graphs=sum(1 for graph in graphs
                 if all(r["eligible"] for r in slice_k if r["graph"]==graph)),
        )
    report=dict(phase=phase,graphs=len(graphs),methods=len(METHODS),
        required_seed_rounds=16,total_seed_trajectories=graph_pairs*16,
        eligible_seed_prefixes=prefixes,
        final_full_coverage=prefixes["16"]["completed_graph_method_prefixes"]==graph_pairs,
        selection="highest verified signed cut encountered in any iteration",
        caveat="Descriptive metadata-only progress; final audit rescales archived partitions")
    atomic_json(out/(phase+"_seed_progress.json"),report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=["pilot","full"],default="full")
    p.add_argument("--root",type=Path,default=ROOT)
    args=p.parse_args()
    print(json.dumps(cumulative(root=args.root,phase=args.phase),indent=2))
