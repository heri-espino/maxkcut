"""Audit G-set feasible-cut references for Experiment023.

Reference cuts are not automatically proved optima. Published upper bounds
are independent and apply only to the documented canonical toroidal graphs.
No solver is called by this module.
"""
from __future__ import annotations
import csv
import hashlib
from functools import lru_cache
from pathlib import Path
from .references import reference_for_graph

DATA=Path("results/gset")
EXP021=DATA/"experiment021-complete-71-gset/statistics/per_instance_methods.csv"
HISTORY=DATA/"experiment022-graph-structure-methods/historical_g63_methods.csv"
GRAPHS=tuple([f"G{i}" for i in range(1,68)]+["G70","G72","G77","G81"])

def _read(path):
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    with path.open(encoding="utf8",newline="") as f: rows=list(csv.DictReader(f))
    if not rows: raise ValueError(f"Empty reference source {path}")
    return rows

@lru_cache(maxsize=4)
def load_references(exp021_path=EXP021,history_path=HISTORY):
    exp021_path,history_path=Path(exp021_path),Path(history_path)
    values={}
    seen=set()
    for r in _read(exp021_path):
        graph=r["graph"]
        key=(graph,r["initializer"])
        if graph not in GRAPHS or key in seen or int(r["restarts"])!=50:
            raise ValueError(f"Unexpected graph, method, or restarts {key}")
        seen.add(key)
        cut=float(r["best_final"])
        if not 0<=cut<float("inf"): raise ValueError(f"Invalid cut {key}")
        if graph not in values or cut>values[graph]["empirical_cut"]:
            values[graph]=dict(empirical_cut=cut,empirical_method=r["initializer"])
    if len(values)!=71 or len(seen)!=639:
        raise ValueError("Expected complete 71 graph x 9 method Experiment021")
    old=[r for r in _read(history_path) if r["graph"]=="G63" and
         r["experiment"]=="013" and r["method"]=="SavedKMeansFlip+KL+ILS"]
    if len(old)!=1 or float(old[0]["best_cut"])!=26917:
        raise ValueError("Historical G63 record missing or changed")
    results={}
    for graph in GRAPHS:
        empirical=values[graph]
        cut=empirical["empirical_cut"]
        provenance="Experiment021 best-of-50 portfolio"
        if graph=="G63" and cut<26917:
            cut=26917.
            provenance="Experiment013 SavedKMeansFlip+KL+ILS"
        paper=reference_for_graph(graph)
        if paper is not None and paper["published_cut"]>cut:
            cut=float(paper["published_cut"])
            provenance=paper["source"]
        upper=paper["published_upper_bound"] if paper else None
        results[graph]=dict(graph=graph,reference_cut=cut,
            reference_kind=("conditional_proven_optimum" if upper is not None
                            and abs(cut-upper)<1e-8 else "best_known_feasible"),
            reference_source=provenance,
            experiment021_best=empirical["empirical_cut"],
            experiment021_winning_initializer=empirical["empirical_method"],
            historical_g63_best=26917. if graph=="G63" else None,
            published_cut=paper["published_cut"] if paper else None,
            published_upper_bound=upper,
            published_reference_status=(paper["reference_status"] if paper else "unavailable"),
            above_published_upper_bound=bool(upper is not None and cut>upper+1e-8))
    meta=dict(experiment021_sha256=hashlib.sha256(exp021_path.read_bytes()).hexdigest(),
              historical_sha256=hashlib.sha256(history_path.read_bytes()).hexdigest(),
              graph_count=len(results),toroidal_paper_count=sum(
                  r["published_cut"] is not None for r in results.values()))
    return results,meta

def compare_cut(cut,ref):
    if cut is None:
        return dict(reference_gap=None,fraction_of_reference=None,
                    reference_assessment="not_measured")
    cut=float(cut)
    if not 0<=cut<float("inf"): raise ValueError("Nonfinite or negative cut")
    known=float(ref["reference_cut"])
    upper=ref["published_upper_bound"]
    assessment=("ABOVE_PUBLISHED_UPPER_BOUND_VERIFY_INPUT" if upper is not None and cut>upper+1e-8
                else "ABOVE_ARCHIVED_REFERENCE_VERIFY_PARTITION" if cut>known+1e-8
                else "MATCHES_REFERENCE" if abs(cut-known)<1e-8 else "BELOW_REFERENCE")
    return dict(reference_gap=known-cut,
                fraction_of_reference=(cut/known if known>0 else None),
                reference_assessment=assessment)
