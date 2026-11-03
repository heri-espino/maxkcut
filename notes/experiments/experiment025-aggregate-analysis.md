# Experiment025 — 71-graph aggregate best-of-16 analysis

**Status:** Provisional numerical analysis of a complete computational sweep; independent NPZ partition rescore and off-VM archive verification remain pending.

- Source input: `full_best_final_by_graph.csv`; SHA256 `8a9d84cfc35f1d424519b862676facc288a6366bc1820e52975a2bf76678de6c`.
- All 1,278 rows are unique, complete_16_best_seen, and have 16 refined starts. The CSV gain, reference-gap, GW-baseline and tied-winner invariants were checked for **all 1,278 rows**.
- Dataset: **71 graphs × 18 protocols × 16 saved starts = 20,448 refinements**, aggregated to best-of-16 and mean-of-16 per graph-method.
- Computed using only aggregate CSV; no EVA solve, no new optimizer run, no access to per-seed partitions or iteration journals.

## 1. Measurement and statistical definition

Let `C[g,m]` be the highest original signed Max-Cut observed over the 16 independently refined starts for graph `g` and method `m`; `G[g] = C[g,gw-rr]`. We report the equal-graph-weighted mean of `100 × (C[g,m] − G[g]) / G[g]` (all observed GW baselines are positive). Absolute mean gain is in original cut-weight units and should not be compared across differently sized graphs without normalization.

A **graph-level paired contrast** compares two methods on each identical graph, not on matched individual GW seeds. The input CSV contains 16-seed means, but no per-seed covariance or paired seed trajectories. All 95% intervals below are percentile bootstrap intervals from **30,000 graph-level resamples**, seed `20261103`; they quantify sensitivity to this graph mix, not uncertainty about the fixed 71-graph census. Unadjusted paired Wilcoxon signed-rank p-values test graph-level best-of-16 cut differences; the GW comparisons are adjusted by Holm for 17 non-GW methods. Graphs in a family may be dependent, so inferential p-values are exploratory.

## 2. Method rankings and relationship to GW

| Rank | Method | Mean gain vs refined GW | 95% graph bootstrap CI | Graphs W / T / L vs GW | Exact tied-winner appearances |
|---:|---|---:|---|---:|---:|
| 1 | `hyperplane-sweep` | +0.318% | [+0.225%, +0.420%] | 59/7/5 | 26 |
| 2 | `control-paired` | +0.272% | [+0.178%, +0.373%] | 55/8/8 | 14 |
| 3 | `em-gibbs` | +0.271% | [+0.196%, +0.352%] | 59/9/3 | 19 |
| 4 | `cutaware-kmeans` | +0.267% | [+0.178%, +0.363%] | 56/5/10 | 15 |
| 5 | `em-hard` | +0.258% | [+0.184%, +0.339%] | 58/9/4 | 18 |
| 6 | `em-cem` | +0.258% | [+0.184%, +0.339%] | 58/9/4 | 18 |
| 7 | `control-tabu` | +0.234% | [+0.149%, +0.326%] | 54/8/9 | 13 |
| 8 | `multigroup-k4` | +0.228% | [+0.146%, +0.316%] | 52/11/8 | 13 |
| 9 | `multigroup-k8` | +0.074% | [+0.032%, +0.127%] | 22/48/1 | 6 |
| 10 | `gw-threshold` | +0.010% | [-0.034%, +0.054%] | 30/18/23 | 5 |
| 11 | `gw-rr` | +0.000% | [+0.000%, +0.000%] | 0/71/0 | 4 |
| 12 | `mst-l2-longest` | +0.000% | [+0.000%, +0.000%] | 0/71/0 | 4 |
| 13 | `mst-l2-balanced` | +0.000% | [+0.000%, +0.000%] | 0/71/0 | 4 |
| 14 | `mst-l2-diameter` | +0.000% | [+0.000%, +0.000%] | 0/71/0 | 4 |
| 15 | `mst-l2-bestcut` | +0.000% | [+0.000%, +0.000%] | 0/71/0 | 4 |
| 16 | `mst-l1-bestcut` | +0.000% | [+0.000%, +0.000%] | 0/71/0 | 4 |
| 17 | `mst-linf-bestcut` | +0.000% | [+0.000%, +0.000%] | 0/71/0 | 4 |
| 18 | `mst-l2-multicut` | -0.000% | [-0.001%, +0.000%] | 0/70/1 | 4 |

- **Leader on this prespecified equal-graph-weighted relative best-of-16 metric:** `hyperplane-sweep` (**+0.318%**, 95% bootstrap [+0.225%, +0.420%]); wins/ties/losses vs refined GW **59/7/5**. Mean absolute cut gain **14.08**.
- **Important:** “graph winner” means the graph-specific best among 18 methods, selected after evaluating all of them. It is an **oracle selector**, not a deployable preselected method; it is automatically at least as good as GW because GW itself is an option.
- There are **44 graphs with a unique winning method**, with the remaining 27 having ties (including three graphs where every method ties). Exact appearances overcount tied wins; fractional winner credit allocates 1/k to each of k tied winners.

## 3. Is Hyperplane Sweep significantly better than other leading methods?

| Opponent | Mean difference, percentage points (hyperplane minus opponent) | Bootstrap 95% CI | Hyperplane graph W/T/L | Paired Wilcoxon p (cuts) |
|---|---:|---|---:|---:|
| `control-paired` | +0.046 | [+0.003, +0.091] | 36/10/25 | 0.226 |
| `em-gibbs` | +0.047 | [-0.011, +0.107] | 36/9/26 | 0.163 |
| `cutaware-kmeans` | +0.051 | [+0.011, +0.092] | 41/7/23 | 0.11 |
| `em-hard` | +0.059 | [+0.001, +0.120] | 37/10/24 | 0.0828 |
| `control-tabu` | +0.084 | [+0.032, +0.138] | 39/7/25 | 0.0313 |
| `multigroup-k4` | +0.090 | [+0.038, +0.142] | 40/12/19 | 0.00039 |
| `gw-rr` | +0.318 | [+0.225, +0.420] | 59/7/5 | 7.7e-11 |

**Interpretation:** Hyperplane Sweep has the best observed equal-graph mean relative gain, but it does **not** significantly dominate the four nearest leading alternatives on two-sided graph-level Wilcoxon signed-rank tests. A bootstrap interval for the *mean relative* contrast can exclude zero while a Wilcoxon *rank test on absolute cut differences* does not; they test different estimands. Do not describe a uniquely statistically proven champion from this one aggregate CSV.

## 4. Stratification by graph family and weight type

| Family | Graphs | Best fixed method by mean relative gain | Gain | Hyperplane Sweep gain |
|---|---:|---|---:|---:|
| random | 30 | `control-tabu` | +0.399% | +0.375% |
| planar | 24 | `hyperplane-sweep` | +0.411% | +0.411% |
| toroidal | 17 | `hyperplane-sweep` | +0.084% | +0.084% |

| Weight type | Graphs | Best fixed method by mean relative gain | Gain | Hyperplane Sweep gain |
|---|---:|---|---:|---:|
| positive | 35 | `cutaware-kmeans` | +0.164% | +0.163% |
| signed | 36 | `hyperplane-sweep` | +0.468% | +0.468% |

**Scientific caveat:** Family-conditional “best method” is selected using the *same test graphs* and is exploratory. Toroidal improvements are substantially smaller, and some methods are worse than GW on some instances. Do not claim an out-of-sample method-selection policy without held-out validation. Family and signedness are confounded in parts of the corpus (e.g., only 3 positive toroidal graphs).

## 5. Oracle winner, previous feasible references, and hard graphs

- Across the 71 graphs, the **graph-specific 18-method oracle selector** improves over GW on **67** graphs and ties on **4**; average equal-graph relative improvement **+0.421%**, mean absolute **+18.08** cut-weight units. This is an in-sample upper envelope over 18 computed methods, not a single reproducible method selection rule.
- Relative to the CSV’s `reference_cut` (archived best-known **feasible** reference, not a certified optimum), the 18-method oracle is **above on 8**, **equal on 13**, and **below on 50** graphs. These are **candidate reference improvements**, not independently verified new public records.

**Graphs provisionally above their archived feasible reference**:

| Graph | Previous reference | New best observed | Gain above reference | Winner(s) |
|---|---:|---:|---:|---|
| G1 | 11,621 | 11,624 | +3 | em-hard, em-cem, em-gibbs |
| G9 | 2,050 | 2,051 | +1 | multigroup-k4 |
| G28 | 3,279 | 3,280 | +1 | control-tabu |
| G36 | 7,657 | 7,661 | +4 | control-paired |
| G40 | 2,379 | 2,384 | +5 | hyperplane-sweep |
| G42 | 2,454 | 2,459 | +5 | hyperplane-sweep |
| G44 | 6,649 | 6,650 | +1 | hyperplane-sweep |
| G53 | 3,843 | 3,844 | +1 | hyperplane-sweep |

**Selected large/hard instances**:

| Graph | Refined GW | Best of 18 | Archived feasible reference | Remaining shortfall |
|---|---:|---:|---:|---:|
| G63 | 26,857 | 26,900 | 26,917 | 17 |
| G70 | 9,536 | 9,556 | 9,558 | 2 |
| G72 | 6,890 | 6,900 | 7,008 | 108 |
| G77 | 9,790 | 9,796 | 9,940 | 144 |
| G81 | 13,836 | 13,842 | 14,060 | 218 |

## 6. Behavioral equivalences and protocol audit flags

- **Six MST-hybrid methods** (`mst-l2-longest`, `mst-l2-balanced`, `mst-l2-diameter`, `mst-l2-bestcut`, `mst-l1-bestcut`, `mst-linf-bestcut`) have **exactly the same 71 graph best-of-16 scores and 16-seed means as refined GW-RR**. The seventh (`mst-l2-multicut`) ties GW on 70 graphs and loses by one cut unit on G54. This is an empirical aggregate equivalence, not proof that the implementation does no work; review per-seed source/journals to determine why no MST improvement survives the shared refinement and GW anchor.
- **EM-Hard and EM-CEM** have **identical 71 graph best-of-16 and mean-of-16 scores**, although reported wall-clock fields can differ. This suggests either performance-equivalent refinements or an implementation/protocol effect; inspect individual seeds and the EM/CEM candidate archives before calling them independent successful strategies.
- Thus the **18 advertised protocols collapse into 11 distinct (best-of-16, mean-of-16) aggregate outcome profiles**. This makes a valuable no-recompute ablation/audit question, not automatically a numerical error.
- The column `wall_seconds` is a last-invocation field, not complete wall time over all 16 refinements. Do not use it to rank computational efficiency.

## 7. Next reproducible steps (without rerunning optimization)

1. Preserve the **entire** local NPZ/JSONL/log/seed tree in a separately verified off-VM backup (file count and SHA256 manifest). Do not reset/clean/pull over the active VM worktree or push raw LFS history.
2. Independently re-open/rescore all saved graph-method winners (ideally all 20,448 seed partitions) against original signed edges and verify graph, factor, partition and optimizer SHA; record exceptions.
3. Export a **compact 20,448-row per-seed statistics table** with graph, family, weight type, method, seed index, verified best cut, initial cut, final cut, source/partition hashes, and success flags. This would support actual 16-seed matched-pair differences, variability, convergence and seed-budget learning curves. Do not upload large NPZs to GitHub.
4. Predefine primary estimand (best-of-16 vs mean-single-seed) and whether per-graph relative gains are the primary scale. Use matched graph/seed comparisons, multiple-comparison control, family/signedness analysis and runtime only if complete timings are available.
5. Consider the MST and EM-CEM redundancies as a scientific diagnostic. Compare raw candidate stage scores vs refined output to see whether the shared Flip/KL/ILS pipeline collapses structurally different initializers.
6. Publish these **aggregate-only** numerical files and analysis through a research PR after verifying the backed-up raw archive and objective rescore. The provided report intentionally does not claim a published record, new approximation guarantee or general superiority outside this fixed G-set.

## Output artifacts

- `experiment025_method_ranking.csv`: one row per method with graph-level gains, intervals, wins/ties/losses, winner credit and Holm-adjusted exploratory tests.
- `experiment025_graph_winners.csv`: all 71 graphs with reference, GW baseline, 18-method oracle, tie information and Hyperplane Sweep score.
- `experiment025_leader_head_to_head.csv`: Hyperplane Sweep versus selected rival methods, graph-level contrasts and p-values.
- Reproduction: run this script against the exact uploaded compact source CSV; the input hash is printed at the top.
