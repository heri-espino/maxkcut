"""Experiment022: retrospective Max-Cut initializer × graph-structure study.

Read only Experiment021's already-completed 71-graph per-instance CSV and
canonical G-set TXT edges; do not solve EVA or repeat any optimization.

Family and +/- edge type labels come from KULeuven-MICAS/openising's
G-set README (accessed 2026-10-29). They are historical benchmark labels,
not inferred from graph drawing coordinates.

Run from the repository root:
    python -m maxkcut.graph_structure_methods
    python -m maxkcut.graph_structure_methods --skip-features
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path("results/gset/experiment021-complete-71-gset/statistics")
INPUT = ROOT / "per_instance_methods.csv"
OUTPUT = Path("results/gset/experiment022-graph-structure-methods")
GENERATOR_LABEL_SOURCE = (
    "https://github.com/KULeuven-MICAS/openising/blob/main/ising/benchmarks/G/README.md"
)
METHODS = (
    "paired", "independent", "sample-antipodal", "sample-pair", "farthest-pair",
    "spherical-pp", "mean-perturb", "pca-antipodal", "gw-screen16",
)


def expand(*ranges, singles=()):
    return set(singles).union(*(set(range(a, b + 1)) for a, b in ranges))


FAMILIES = {
    "random": expand((1, 10), (22, 31), (43, 47), singles=(55, 56, 60, 61, 70)),
    "planar": expand((14, 21), (35, 42), (51, 54), singles=(58, 59, 63, 64)),
    "toroidal": {11, 12, 13, 32, 33, 34, 48, 49, 50, 57, 62, 65, 66, 67, 72, 77, 81},
}
POSITIVE_ONLY = expand(
    (1, 5), (14, 17), (22, 26), (35, 38), (43, 47), (51, 54),
    singles=(48, 49, 50, 55, 58, 60, 63, 70),
)


def validate_classification():
    expected = set(range(1, 68)) | {70, 72, 77, 81}
    labeled = set().union(*FAMILIES.values())
    assert labeled == expected and sum(map(len, FAMILIES.values())) == 71
    assert POSITIVE_ONLY <= labeled
    assert all(not (FAMILIES[a] & FAMILIES[b]) for a in FAMILIES for b in FAMILIES if a < b)
    return expected


def labels(graph):
    num = int(graph.removeprefix("G"))
    family = next(name for name, ids in FAMILIES.items() if num in ids)
    weight_type = "positive" if num in POSITIVE_ONLY else "signed"
    return family, weight_type


def load_results(path=INPUT):
    validate_classification()
    scores = defaultdict(dict)
    expected_graphs = {f"G{i}" for i in validate_classification()}
    with Path(path).open(newline="", encoding="utf8") as f:
        for row in csv.DictReader(f):
            g, m = row["graph"], row["initializer"]
            if g not in expected_graphs:
                raise ValueError(f"Unexpected graph {g}")
            if m not in METHODS or m in scores[g]:
                raise ValueError(f"Unknown/duplicate method: {g} {m}")
            if int(row["restarts"]) != 50:
                raise ValueError(f"Expected 50 restarts for {g} {m}")
            scores[g][m] = {
                "best": float(row["best_final"]),
                "mean": float(row["mean_final"]),
                "seconds": float(row["mean_seconds"]),
                "extra_eval": int(row["additional_initial_cut_evaluations"]),
            }
    if len(scores) != 71 or any(set(v) != set(METHODS) for v in scores.values()):
        raise ValueError("Expected complete 71 graph × 9 initializer experiment")
    if any(v["gw-screen16"]["extra_eval"] != 16 for v in scores.values()):
        raise ValueError("GW screen-16 budget metadata mismatch")
    return dict(sorted(scores.items(), key=lambda kv: int(kv[0][1:])))


def graph_features(path):
    """Exact undirected adjacency features; no spatial coordinates assumed."""
    path = Path(path)
    with path.open(encoding="utf8") as f:
        n, declared_m = map(int, f.readline().split()[:2])
        neighbors = [set() for _ in range(n)]
        weights = {}
        nnegative = 0
        for line in f:
            if not line.strip():
                continue
            i, j, w = line.split()[:3]
            i, j, w = int(i)-1, int(j)-1, float(w)
            if not (0 <= i < n and 0 <= j < n) or i == j:
                raise ValueError(f"Bad edge in {path}: {line}")
            e = (min(i, j), max(i, j))
            if e in weights:
                raise ValueError(f"Duplicate edge in {path}: {line}")
            weights[e] = w
            neighbors[i].add(j)
            neighbors[j].add(i)
            nnegative += w < 0
    m = len(weights)
    if m != declared_m:
        raise ValueError(f"Edge count mismatch {path}: {m} != {declared_m}")
    degree = [len(s) for s in neighbors]
    avg_degree = 2*m/n
    cv = (sum((x-avg_degree)**2 for x in degree)/n)**0.5 / max(avg_degree, 1e-12)
    wedge_count = sum(d*(d-1)//2 for d in degree)
    # Count each triangle at its smallest vertex exactly once.
    triangles = 0
    for i in range(n):
        ni = neighbors[i]
        higher = [j for j in ni if j > i]
        for j in higher:
            triangles += len({k for k in ni.intersection(neighbors[j]) if k > j})
    transitivity = 3*triangles/wedge_count if wedge_count else 0.0
    visited, components = set(), 0
    largest = 0
    for i in range(n):
        if i in visited:
            continue
        components += 1
        stack, count = [i], 0
        visited.add(i)
        while stack:
            v = stack.pop()
            count += 1
            for u in neighbors[v]:
                if u not in visited:
                    visited.add(u)
                    stack.append(u)
        largest = max(largest, count)
    return dict(graph=path.stem, graph_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), n=n, m=m, average_degree=avg_degree,
                degree_cv=cv, triangles=triangles, transitivity=transitivity,
                negative_edge_fraction=nnegative/m if m else 0.0,
                components=components, largest_component_fraction=largest/n,
                cycle_rank=m-n+components, normalized_cycle_rank=(m-n+components)/max(m, 1))


def winner_tables(scores):
    graph_rows = []
    method_rows = []
    for g, record in scores.items():
        family, weight_type = labels(g)
        top = max(x["best"] for x in record.values())
        worst = min(x["best"] for x in record.values())
        winners = [m for m in METHODS if record[m]["best"] == top]
        base = record["paired"]
        graph_rows.append(dict(graph=g, family=family, weight_type=weight_type,
                               informative=int(top > worst), top_best=top,
                               number_tied=len(winners), winning_methods=";".join(winners)))
        for m in METHODS:
            cur = record[m]
            method_rows.append(dict(graph=g, family=family, weight_type=weight_type,
                method=m, best=cur["best"], mean=cur["mean"],
                best_gain_vs_paired=cur["best"]-base["best"],
                mean_gain_vs_paired=cur["mean"]-base["mean"],
                relative_best_gain=(cur["best"]-base["best"])/max(abs(base["best"]), 1.0),
                shared_win=int(m in winners), outright_win=int(m in winners and len(winners)==1),
                fractional_win=(1/len(winners) if m in winners else 0),
                extra_initial_cut_evaluations=cur["extra_eval"]))
    return graph_rows, method_rows


def aggregate_methods(rows):
    out = []
    for fam in FAMILIES:
        for method in METHODS:
            rr = [r for r in rows if r["family"] == fam and r["method"] == method]
            out.append(dict(family=fam, method=method, graphs=len(rr),
                shared_wins=sum(r["shared_win"] for r in rr),
                outright_wins=sum(r["outright_win"] for r in rr),
                fractional_wins=sum(r["fractional_win"] for r in rr),
                mean_cut_difference=sum(r["mean_gain_vs_paired"] for r in rr)/len(rr),
                mean_best_difference=sum(r["best_gain_vs_paired"] for r in rr)/len(rr)))
    return out


def loo_selection(scores, exclude_screen16=False):
    """Disjoint train/test graph IDs; family-only vs global selection.

    Optimize training mean of relative best-of-50 gain over paired.
    This is predictive RETROSPECTIVE evaluation, not equal-wall-clock validation.
    """
    methods = tuple(m for m in METHODS if not (exclude_screen16 and m == "gw-screen16"))
    res = []
    for held, observed in scores.items():
        family, weight_type = labels(held)
        train = [(g, rr) for g, rr in scores.items() if g != held]
        for policy in ("global", "family"):
            subset = train if policy == "global" else [(g, rr) for g, rr in train if labels(g)[0] == family]
            def train_score(method):
                return sum((rr[method]["best"] - rr["paired"]["best"])
                    / max(abs(rr["paired"]["best"]), 1.0) for _, rr in subset) / len(subset)
            selection = min(methods, key=lambda m: (-train_score(m), METHODS.index(m)))
            chosen = observed[selection]["best"]
            oracle = max(observed[m]["best"] for m in methods)
            baseline = observed["paired"]["best"]
            res.append(dict(graph=held, family=family, weight_type=weight_type,
                screen16_allowed=int(not exclude_screen16), policy=policy,
                selected_method=selection, selected_cut=chosen,
                paired_cut=baseline, gain=chosen-baseline,
                relative_gain=(chosen-baseline)/max(abs(baseline), 1.0),
                regret=oracle-chosen))
    return res


def fisher_two_sided(a, b, c, d):
    """Exact conditional Fisher test; exploratory only, ignores family dependence."""
    row1, row2, col1 = a+b, c+d, a+c
    total = row1+row2
    lower = max(0, col1-row2)
    upper = min(row1, col1)
    def p(x):
        return math.comb(col1, x)*math.comb(total-col1, row1-x)/math.comb(total,row1)
    observed = p(a)
    return sum(p(x) for x in range(lower,upper+1) if p(x) <= observed+1e-12)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf8") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def execute(input_csv=INPUT, output=OUTPUT, data_dir=Path("data"), *, skip_features=False):
    scores=load_results(input_csv)
    output=Path(output)
    output.mkdir(parents=True, exist_ok=True)
    graph_rows, method_rows=winner_tables(scores)
    agg=aggregate_methods(method_rows)
    for file, rows in (("graph_winners.csv", graph_rows),
                       ("method_by_family.csv", agg),
                       ("method_by_graph.csv", method_rows)):
        write_csv(output/file, rows)
    all_loo=[]
    for excluded in (False, True):
        all_loo.extend(loo_selection(scores,exclude_screen16=excluded))
    write_csv(output/"leave_one_graph_out.csv", all_loo)
    features=[]
    if not skip_features:
        for g in scores:
            row=graph_features(Path(data_dir)/f"{g}.txt")
            row.update(zip(("family", "weight_type"), labels(g)))
            expected_signed = labels(g)[1] == "signed"
            if expected_signed != (row["negative_edge_fraction"]>0):
                raise ValueError(f"External weight class disagrees with original signed edges for {g}")
            features.append(row)
        write_csv(output/"graph_features.csv", features)
    counts = {}
    for fam in FAMILIES:
        group=[r for r in graph_rows if r["family"]==fam]
        counts[fam]=len(group)
    def cv_summary(exclude, policy):
        rr=[r for r in all_loo if r["screen16_allowed"]==int(not exclude) and r["policy"]==policy]
        return dict(mean_gain=sum(r["gain"] for r in rr)/len(rr),
                    mean_relative_gain_percent=100*sum(r["relative_gain"] for r in rr)/len(rr),
                    mean_regret=sum(r["regret"] for r in rr)/len(rr),
                    gains=sum(r["gain"]>0 for r in rr),
                    losses=sum(r["gain"]<0 for r in rr))
    gwrows=[r for r in method_rows if r["method"]=="gw-screen16"]
    toro=[r for r in gwrows if r["family"]=="toroidal" and r["weight_type"]=="signed"]
    other=[r for r in gwrows if r["family"]!="toroidal" and r["weight_type"]=="signed"]
    table=[sum(r["shared_win"] for r in toro), len(toro)-sum(r["shared_win"] for r in toro),
           sum(r["shared_win"] for r in other), len(other)-sum(r["shared_win"] for r in other)]
    report=dict(status="retrospective_analysis", input_csv_sha256=hashlib.sha256(Path(input_csv).read_bytes()).hexdigest(), groups=counts, total_graphs=len(scores),
        total_methods=len(METHODS), restarts=50, external_label_source=GENERATOR_LABEL_SOURCE,
        caveat="No equal-budget comparison; 16 additional initial cut evaluations for GW screen-16",
        fisher_signed_toroidal_vs_other_signed_gw_shared_win_table=table,
        fisher_two_sided_exploratory=fisher_two_sided(*table),
        loo={f"{policy}_{'without_screen16' if exclude else 'all_methods'}":cv_summary(exclude,policy)
             for exclude in (False,True) for policy in ("global","family")},
        output_files=["graph_winners.csv","method_by_family.csv","method_by_graph.csv",
                      "leave_one_graph_out.csv"]+([] if skip_features else ["graph_features.csv"]))
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n", encoding="utf8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=INPUT)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--data-dir",type=Path,default=Path("data"))
    parser.add_argument("--skip-features",action="store_true")
    args=parser.parse_args()
    print(json.dumps(execute(args.input,args.output,args.data_dir,skip_features=args.skip_features),indent=2))


if __name__=="__main__":
    main()