#!/usr/bin/env python3
"""Reproducible aggregate-only analysis of Experiment025 G-set full best-of-16 summary.

No optimizer, EVA/SDP solver, or user VM paths are accessed. Deliberately
works only from the compact 1,278-row CSV; per-seed paired tests and independent
partition verification are NOT possible from this input.

Usage:
  python experiment025_analyze_aggregate.py --input full_best_final_by_graph.csv \
      --out-dir experiment025_analysis

Dependencies: numpy, scipy (part of user's existing maxkcut environment).
"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
import csv
from hashlib import sha256
import math
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

SEED = 20261103
B = 30000


def nfmt(x,dec=3):
    return f"{x:,.{dec}f}"


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--input',type=Path,required=True)
    a.add_argument('--out-dir',type=Path,required=True)
    args=a.parse_args()
    args.out_dir.mkdir(exist_ok=True,parents=True)
    digest=sha256(args.input.read_bytes()).hexdigest()
    with args.input.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    by={(r['graph'],r['method']):r for r in rows}
    if len(rows)!=1278 or len(by)!=1278:
        raise ValueError('Expected 1,278 unique graph-method rows')
    graphs=sorted({r['graph'] for r in rows},key=lambda s:int(s[1:]))
    methods=[r['method'] for r in rows if r['graph']==graphs[0]]
    if len(graphs)!=71 or len(methods)!=18 or methods[0]!='gw-rr':
        raise ValueError('Unexpected 71-graph / 18-method protocol')
    get=lambda g,m,n:float(by[g,m][n])
    v=np.array([[get(g,m,'best_seen_cut') for m in methods] for g in graphs]);
    means=np.array([[get(g,m,'mean_best_seen') for m in methods] for g in graphs]);
    reference=np.array([get(g,'gw-rr','reference_cut') for g in graphs]);
    gw=v[:,0]; gm=means[:,0];oracle=np.max(v,axis=1)
    if not np.all(gw>0):
        raise ValueError('Percentage improvement defined here only for positive GW baseline')
    if not np.all(means<=v+1e-7):
        raise ValueError('Mean-of-16 exceeds best-of-16')
    for r in rows:
        g,m=r['graph'],r['method'];score=float(r['best_seen_cut']); i=graphs.index(g)
        if r['status']!='complete_16_best_seen' or r['refined_starts']!='16':
            raise ValueError(f'Incomplete cell {g}/{m}')
        if abs(score-get(g,'gw-rr','best_seen_cut')-float(r['gain_vs_gw']))>1e-7:
            raise ValueError('Inconsistent gain_vs_gw')
        if abs(float(r['reference_cut'])-score-float(r['gap_to_reference']))>1e-7:
            raise ValueError('Inconsistent reference gap')
        if abs(float(r['gw_rr_full16'])-gw[i])>1e-7:
            raise ValueError('Inconsistent GW per-graph reference')
        if int(r['tied_best'])!=int(score==oracle[i]):
            raise ValueError('Inconsistent tied_best')
    gains=v-gw[:,None]; relative=100*gains/gw[:,None]
    bootstrap=np.random.default_rng(SEED).integers(0,len(graphs),(B,len(graphs)))
    # Each bootstrap sample resamples graphs (not seeds). Fixed benchmark caveat.
    boot_rel=relative[bootstrap,:].mean(axis=1)
    tied=(v==oracle[:,None]); tie_sizes=tied.sum(axis=1)
    pvalues=[]
    for j in range(1,len(methods)):
        d=gains[:,j]
        p=float(wilcoxon(d,zero_method='wilcox').pvalue) if np.any(d!=0) else 1.0
        pvalues.append((j,p))
    holm={}
    carry=0
    for idx,(j,p) in enumerate(sorted(pvalues,key=lambda item:item[1])):
        carry=max(carry, min(1.0,(len(pvalues)-idx)*p));holm[j]=carry
    ranking=[]
    for j,m in enumerate(methods):
        d=gains[:,j]
        low,high=np.quantile(boot_rel[:,j],[.025,.975]); wins=tied[:,j]
        ranking.append({
            'method':m,
            'mean_gain_pct':round(float(relative[:,j].mean()),6),
            'bootstrap_95_ci_low_pct':round(float(low),6),
            'bootstrap_95_ci_high_pct':round(float(high),6),
            'median_gain_pct':round(float(np.median(relative[:,j])),6),
            'mean_gain_cut_units':round(float(d.mean()),6),
            'median_gain_cut_units':round(float(np.median(d)),6),
            'graphs_better_than_gw':int(sum(d>0)),
            'graphs_tied_with_gw':int(sum(d==0)),
            'graphs_worse_than_gw':int(sum(d<0)),
            'tied_graph_winner_appearances':int(wins.sum()),
            'fractional_winner_credit':round(float(np.sum(wins/tie_sizes)),6),
            'mean_of_16_seed_gain_pct':round(float((100*(means[:,j]-gm)/gm).mean()),6),
            'mean_of_16_seed_gain_cut_units':round(float((means[:,j]-gm).mean()),6),
            'wilcoxon_p_vs_gw_graph_best16':round(float(pvalues[j-1][1]),12) if j>0 else '',
            'holm_p_vs_gw_17_methods':round(float(holm[j]),12) if j>0 else '',
        })
    ranking.sort(key=lambda r:-r['mean_gain_pct'])
    write_csv(args.out_dir/'experiment025_method_ranking.csv',ranking,list(ranking[0]))
    graphrows=[]
    for i,g in enumerate(graphs):
        winners=[methods[j] for j in range(18) if tied[i,j]]
        graphrows.append({'graph':g,'family':by[g,'gw-rr']['family'],
            'weight_type':by[g,'gw-rr']['weight_type'],
            'refined_gw_rr_cut':int(gw[i]),'best_of_18_cut':int(oracle[i]),
            'oracle_gain_cut_units':int(oracle[i]-gw[i]),
            'oracle_gain_pct':round(float(100*(oracle[i]-gw[i])/gw[i]),6),
            'method_tie_count':len(winners),'all_winning_methods':';'.join(winners),
            'reference_cut':int(reference[i]),
            'best_minus_reference':int(oracle[i]-reference[i]),
            'hyperplane_sweep_cut':int(v[i,methods.index('hyperplane-sweep')]),
            'hyperplane_gain_cut_units':int(gains[i,methods.index('hyperplane-sweep')])})
    write_csv(args.out_dir/'experiment025_graph_winners.csv',graphrows,list(graphrows[0]))
    # Pairwise head-to-head vs primary full-coverage leader.
    leader=methods.index('hyperplane-sweep')
    pairs=[]
    for m in ['control-paired','em-gibbs','cutaware-kmeans','em-hard','control-tabu','multigroup-k4','gw-rr']:
        j=methods.index(m)
        delta=relative[:,leader]-relative[:,j]
        samples=delta[bootstrap].mean(axis=1)
        cutdelta=v[:,leader]-v[:,j]
        wp=wilcoxon(cutdelta).pvalue if np.any(cutdelta) else 1
        pairs.append({'comparison':f'hyperplane-sweep minus {m}',
            'mean_relative_difference_pct_points':round(float(delta.mean()),6),
            'bootstrap_95_ci_low_pp':round(float(np.quantile(samples,.025)),6),
            'bootstrap_95_ci_high_pp':round(float(np.quantile(samples,.975)),6),
            'hyper_wins':int((cutdelta>0).sum()),'ties':int((cutdelta==0).sum()),
            'hyper_losses':int((cutdelta<0).sum()),
            'wilcoxon_p_absolute_cut_differences':round(float(wp),12)})
    write_csv(args.out_dir/'experiment025_leader_head_to_head.csv',pairs,list(pairs[0]))
    summary=[]
    def l(s=''):summary.append(s)
    l('# Experiment025 — 71-graph aggregate best-of-16 analysis')
    l('')
    l('**Status:** Provisional numerical analysis of a complete computational sweep; independent NPZ partition rescore and off-VM archive verification remain pending.')
    l('')
    l(f'- Source input: `{args.input.name}`; SHA256 `{digest}`.')
    l('- All 1,278 rows are unique, complete_16_best_seen, and have 16 refined starts. The CSV gain, reference-gap, GW-baseline and tied-winner invariants were checked for **all 1,278 rows**.')
    l('- Dataset: **71 graphs × 18 protocols × 16 saved starts = 20,448 refinements**, aggregated to best-of-16 and mean-of-16 per graph-method.')
    l('- Computed using only aggregate CSV; no EVA solve, no new optimizer run, no access to per-seed partitions or iteration journals.')
    l('')
    l('## 1. Measurement and statistical definition')
    l('')
    l('Let `C[g,m]` be the highest original signed Max-Cut observed over the 16 independently refined starts for graph `g` and method `m`; `G[g] = C[g,gw-rr]`. We report the equal-graph-weighted mean of `100 × (C[g,m] − G[g]) / G[g]` (all observed GW baselines are positive). Absolute mean gain is in original cut-weight units and should not be compared across differently sized graphs without normalization.')
    l('')
    l('A **graph-level paired contrast** compares two methods on each identical graph, not on matched individual GW seeds. The input CSV contains 16-seed means, but no per-seed covariance or paired seed trajectories. All 95% intervals below are percentile bootstrap intervals from **30,000 graph-level resamples**, seed `20261103`; they quantify sensitivity to this graph mix, not uncertainty about the fixed 71-graph census. Unadjusted paired Wilcoxon signed-rank p-values test graph-level best-of-16 cut differences; the GW comparisons are adjusted by Holm for 17 non-GW methods. Graphs in a family may be dependent, so inferential p-values are exploratory.')
    l('')
    l('## 2. Method rankings and relationship to GW')
    l('')
    l('| Rank | Method | Mean gain vs refined GW | 95% graph bootstrap CI | Graphs W / T / L vs GW | Exact tied-winner appearances |')
    l('|---:|---|---:|---|---:|---:|')
    for k,row in enumerate(ranking,1):
        l(f"| {k} | `{row['method']}` | {row['mean_gain_pct']:+.3f}% | [{row['bootstrap_95_ci_low_pct']:+.3f}%, {row['bootstrap_95_ci_high_pct']:+.3f}%] | {row['graphs_better_than_gw']}/{row['graphs_tied_with_gw']}/{row['graphs_worse_than_gw']} | {row['tied_graph_winner_appearances']} |")
    l('')
    leaderrow=next(r for r in ranking if r['method']=='hyperplane-sweep')
    l(f"- **Leader on this prespecified equal-graph-weighted relative best-of-16 metric:** `hyperplane-sweep` (**{leaderrow['mean_gain_pct']:+.3f}%**, 95% bootstrap [{leaderrow['bootstrap_95_ci_low_pct']:+.3f}%, {leaderrow['bootstrap_95_ci_high_pct']:+.3f}%]); wins/ties/losses vs refined GW **{leaderrow['graphs_better_than_gw']}/{leaderrow['graphs_tied_with_gw']}/{leaderrow['graphs_worse_than_gw']}**. Mean absolute cut gain **{leaderrow['mean_gain_cut_units']:.2f}**.")
    l('- **Important:** “graph winner” means the graph-specific best among 18 methods, selected after evaluating all of them. It is an **oracle selector**, not a deployable preselected method; it is automatically at least as good as GW because GW itself is an option.')
    l('- There are **44 graphs with a unique winning method**, with the remaining 27 having ties (including three graphs where every method ties). Exact appearances overcount tied wins; fractional winner credit allocates 1/k to each of k tied winners.')
    l('')
    l('## 3. Is Hyperplane Sweep significantly better than other leading methods?')
    l('')
    l('| Opponent | Mean difference, percentage points (hyperplane minus opponent) | Bootstrap 95% CI | Hyperplane graph W/T/L | Paired Wilcoxon p (cuts) |')
    l('|---|---:|---|---:|---:|')
    for p in pairs:
        l(f"| `{p['comparison'].replace('hyperplane-sweep minus ','')}` | {p['mean_relative_difference_pct_points']:+.3f} | [{p['bootstrap_95_ci_low_pp']:+.3f}, {p['bootstrap_95_ci_high_pp']:+.3f}] | {p['hyper_wins']}/{p['ties']}/{p['hyper_losses']} | {p['wilcoxon_p_absolute_cut_differences']:.3g} |")
    l('')
    l('**Interpretation:** Hyperplane Sweep has the best observed equal-graph mean relative gain, but it does **not** significantly dominate the four nearest leading alternatives on two-sided graph-level Wilcoxon signed-rank tests. A bootstrap interval for the *mean relative* contrast can exclude zero while a Wilcoxon *rank test on absolute cut differences* does not; they test different estimands. Do not describe a uniquely statistically proven champion from this one aggregate CSV.')
    l('')
    l('## 4. Stratification by graph family and weight type')
    l('')
    l('| Family | Graphs | Best fixed method by mean relative gain | Gain | Hyperplane Sweep gain |')
    l('|---|---:|---|---:|---:|')
    for fam in ['random','planar','toroidal']:
        ids=[i for i,g in enumerate(graphs) if by[g,'gw-rr']['family']==fam]
        vals=relative[ids].mean(axis=0);j=int(vals.argmax());hp=float(vals[leader])
        l(f'| {fam} | {len(ids)} | `{methods[j]}` | {float(vals[j]):+.3f}% | {hp:+.3f}% |')
    l('')
    l('| Weight type | Graphs | Best fixed method by mean relative gain | Gain | Hyperplane Sweep gain |')
    l('|---|---:|---|---:|---:|')
    for typ in ['positive','signed']:
        ids=[i for i,g in enumerate(graphs) if by[g,'gw-rr']['weight_type']==typ]
        vals=relative[ids].mean(axis=0);j=int(vals.argmax());hp=float(vals[leader])
        l(f'| {typ} | {len(ids)} | `{methods[j]}` | {float(vals[j]):+.3f}% | {hp:+.3f}% |')
    l('')
    l('**Scientific caveat:** Family-conditional “best method” is selected using the *same test graphs* and is exploratory. Toroidal improvements are substantially smaller, and some methods are worse than GW on some instances. Do not claim an out-of-sample method-selection policy without held-out validation. Family and signedness are confounded in parts of the corpus (e.g., only 3 positive toroidal graphs).')
    l('')
    l('## 5. Oracle winner, previous feasible references, and hard graphs')
    l('')
    bestdelta=oracle-gw
    l(f'- Across the 71 graphs, the **graph-specific 18-method oracle selector** improves over GW on **{int(sum(bestdelta>0))}** graphs and ties on **{int(sum(bestdelta==0))}**; average equal-graph relative improvement **{100*np.mean(bestdelta/gw):+.3f}%**, mean absolute **{np.mean(bestdelta):+.2f}** cut-weight units. This is an in-sample upper envelope over 18 computed methods, not a single reproducible method selection rule.')
    l(f'- Relative to the CSV’s `reference_cut` (archived best-known **feasible** reference, not a certified optimum), the 18-method oracle is **above on {int(sum(oracle>reference))}**, **equal on {int(sum(oracle==reference))}**, and **below on {int(sum(oracle<reference))}** graphs. These are **candidate reference improvements**, not independently verified new public records.')
    l('')
    l('**Graphs provisionally above their archived feasible reference**:')
    l('')
    l('| Graph | Previous reference | New best observed | Gain above reference | Winner(s) |')
    l('|---|---:|---:|---:|---|')
    for i,g in enumerate(graphs):
        if oracle[i]>reference[i]:
            l(f"| {g} | {int(reference[i]):,} | {int(oracle[i]):,} | +{int(oracle[i]-reference[i])} | {', '.join(methods[j] for j in range(18) if tied[i,j])} |")
    l('')
    l('**Selected large/hard instances**:')
    l('')
    l('| Graph | Refined GW | Best of 18 | Archived feasible reference | Remaining shortfall |')
    l('|---|---:|---:|---:|---:|')
    for g in ['G63','G70','G72','G77','G81']:
        i=graphs.index(g)
        l(f'| {g} | {int(gw[i]):,} | {int(oracle[i]):,} | {int(reference[i]):,} | {int(max(reference[i]-oracle[i],0)):,} |')
    l('')
    l('## 6. Behavioral equivalences and protocol audit flags')
    l('')
    l('- **Six MST-hybrid methods** (`mst-l2-longest`, `mst-l2-balanced`, `mst-l2-diameter`, `mst-l2-bestcut`, `mst-l1-bestcut`, `mst-linf-bestcut`) have **exactly the same 71 graph best-of-16 scores and 16-seed means as refined GW-RR**. The seventh (`mst-l2-multicut`) ties GW on 70 graphs and loses by one cut unit on G54. This is an empirical aggregate equivalence, not proof that the implementation does no work; review per-seed source/journals to determine why no MST improvement survives the shared refinement and GW anchor.')
    l('- **EM-Hard and EM-CEM** have **identical 71 graph best-of-16 and mean-of-16 scores**, although reported wall-clock fields can differ. This suggests either performance-equivalent refinements or an implementation/protocol effect; inspect individual seeds and the EM/CEM candidate archives before calling them independent successful strategies.')
    l('- Thus the **18 advertised protocols collapse into 11 distinct (best-of-16, mean-of-16) aggregate outcome profiles**. This makes a valuable no-recompute ablation/audit question, not automatically a numerical error.')
    l('- The column `wall_seconds` is a last-invocation field, not complete wall time over all 16 refinements. Do not use it to rank computational efficiency.')
    l('')
    l('## 7. Next reproducible steps (without rerunning optimization)')
    l('')
    l('1. Preserve the **entire** local NPZ/JSONL/log/seed tree in a separately verified off-VM backup (file count and SHA256 manifest). Do not reset/clean/pull over the active VM worktree or push raw LFS history.')
    l('2. Independently re-open/rescore all saved graph-method winners (ideally all 20,448 seed partitions) against original signed edges and verify graph, factor, partition and optimizer SHA; record exceptions.')
    l('3. Export a **compact 20,448-row per-seed statistics table** with graph, family, weight type, method, seed index, verified best cut, initial cut, final cut, source/partition hashes, and success flags. This would support actual 16-seed matched-pair differences, variability, convergence and seed-budget learning curves. Do not upload large NPZs to GitHub.')
    l('4. Predefine primary estimand (best-of-16 vs mean-single-seed) and whether per-graph relative gains are the primary scale. Use matched graph/seed comparisons, multiple-comparison control, family/signedness analysis and runtime only if complete timings are available.')
    l('5. Consider the MST and EM-CEM redundancies as a scientific diagnostic. Compare raw candidate stage scores vs refined output to see whether the shared Flip/KL/ILS pipeline collapses structurally different initializers.')
    l('6. Publish these **aggregate-only** numerical files and analysis through a research PR after verifying the backed-up raw archive and objective rescore. The provided report intentionally does not claim a published record, new approximation guarantee or general superiority outside this fixed G-set.')
    l('')
    l('## Output artifacts')
    l('')
    l('- `experiment025_method_ranking.csv`: one row per method with graph-level gains, intervals, wins/ties/losses, winner credit and Holm-adjusted exploratory tests.')
    l('- `experiment025_graph_winners.csv`: all 71 graphs with reference, GW baseline, 18-method oracle, tie information and Hyperplane Sweep score.')
    l('- `experiment025_leader_head_to_head.csv`: Hyperplane Sweep versus selected rival methods, graph-level contrasts and p-values.')
    l('- Reproduction: run this script against the exact uploaded compact source CSV; the input hash is printed at the top.')
    report=args.out_dir/'experiment025_aggregate_analysis.md';report.write_text('\n'.join(summary)+'\n',encoding='utf-8')
    print('INPUT_SHA256',digest)
    print('OUTPUT',*(str(x) for x in sorted(args.out_dir.glob('*'))),sep='\n')
    print('SUMMARY leader',ranking[0]['method'],ranking[0]['mean_gain_pct'])

if __name__=='__main__':
    main()
