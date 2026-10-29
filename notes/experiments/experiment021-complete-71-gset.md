# Experiment021 — Completion of all 71 canonical G-set Max-Cut instances

## Status and research objective

**COMPLETE, verified 2026-10-29.** The [self-hosted Windows GitHub Actions run 40831249367](https://github.com/heri-espino/maxkcut/actions/runs/40831249367) finished successfully: all tests, all three missing nonconvex EVA factors, 1,350 additional searches, strict cross-graph validation and uploads passed.

Goal: close the remaining **G60, G61, G62** gap in the completed 68/71 Experiment020 study without recomputing or rewriting its 30,600 historical searches. This is a separate, append-only research experiment. Original Experiment020 records remain at `results/gset/experiment020-all-gset-paired-inits/` and G63 retains Experiment019's verified 450-run source.

## Scope and reproducibility invariants

- Canonical input: 71 graphs from `data/G*.txt`, including G1–G67, G70, G72, G77, G81; original signed weights and graph identifiers.
- Existing analyses: 68 graphs × 9 methods × 50 starts = **30,600 preserved runs**, including reused G63.
- New analyses: **3 graphs × 9 methods × 50 starts = 1,350 additional runs**.
- **Final strict output: 71 graphs, 31,950 individually stored complete searches**, matched on restart IDs 0–49 and downstream optimization seeds.
- Newly generated EVA factors: `results/gset/experiment021-complete-71-gset/eva-numba/{cases,factors,archives,logs,trials}/`. Existing source factors from Experiment002 and Experiment003 remain untouched.
- New cut searches: `results/gset/experiment021-complete-71-gset/{G60,G61,G62}/`, with each method/restart's intermediate scores, partitions and refinement traces.
- Consolidated long-format dataset and statistics: `results/gset/experiment021-complete-71-gset/statistics/{report.json,coverage.csv,all_runs.csv,per_instance_methods.csv,paired_vs_original_kmeansnm.csv}`.
- A standalone inventory maps every graph to the **original** factor and result location. Historical searches are not copied into new graph directories or re-executed.
- Verification includes graph SHA256, exact factor SHA256, original factor optimizer metadata, 450-case summaries for each graph and a strict final gate on all 71 graphs and 31,950 rows.
- Software: existing `maxkcut` Conda Python, Numba/NumPy/SciPy, Windows 11, self-hosted 32-logical-processor runner. Code revision and runner are recoverable from the successful workflow and case metadata.

### EVA generation

Generate missing G60/G61/G62 factors using exactly the same Numba sequential EVA family and nominal settings as original Experiment003: seed **2026**, tolerance **0.001 radians**, up to 10,000 sweeps, automatic rank `ceil(sqrt(2(n+1)))`, one BLAS thread, 25-sweep checkpoint intervals, 20 initial rounding restarts. Each graph has 7,000 vertices; new EVA reached convergence:

| Graph | Vertices | Edges | EVA sweeps | Status |
| --- | ---: | ---: | ---: | --- |
| G60 | 7,000 | 17,148 | 239 | converged |
| G61 | 7,000 | 17,148 | 235 | converged |
| G62 | 7,000 | 14,000 | 571 | converged |

These are **nonconvex rank-constrained vector-relaxation factors**, not solutions or certificates of the full convex Max-Cut SDP. Convergence reports are internal EVA angular tolerance conditions, not proofs of global optimality. Each factor was saved and validated against its original signed graph.

### Matched search protocol

All 71 graphs use the same Experiment020 protocol: nine initializer families (`paired` control, `independent`, `sample-antipodal`, `sample-pair`, `farthest-pair`, `spherical-pp`, `mean-perturb`, `pca-antipodal`, `gw-screen16`), 50 restarts each, spherical Lloyd with up to 100 iterations retaining the best actual cut, positive flips (20 sweeps), generalized KL (30 passes), and 80-cycle ILS (8 inner KL passes; kicks 8, 16, 32, 64; temperature 12). 32 independent processes execute search tasks.

Exact seed protocol for graph number `g`, restart `r`:
- K-means initializer `SeedSequence([2026,g,2,r,0])` for original paired; `[...,1]` for alternative initializer streams.
- Flips `[2026,g,r,1313]`, KL `[2026,g,r,101]`, ILS `[2026,g,r,1401]`.
- GW screen-16 expends 16 additional starting-cut evaluations and PCA initialization has preprocessing costs; **comparing the maximum across nine families to one family is not an equal-budget comparison**.

## Results

### Newly completed graphs: best observed signed Max-Cut after full pipeline

| Graph | Winning initialization | Best of nine | Original K-MeansNM best | Best-of-nine gain |
| --- | --- | ---: | ---: | ---: |
| G60 | sample-antipodal | 14,124 | 14,103 | +21 |
| G61 | independent | 5,721 | 5,718 | +3 |
| G62 | mean-perturb | 4,816 | 4,808 | +8 |

The new benchmark portfolio improves the observed best-of-50 control score on **58 of 71 graphs**, ties it on the other 13, and never loses *by construction* because the portfolio contains the control. It does not imply that each alternative improves mean quality; selection across nine methods expends substantially more computational effort.

### Graph-level paired mean differences versus original K-MeansNM

The following is the **unweighted mean of 71 graph-specific differences in average final cut** (50 matched restart IDs per graph). Raw cuts from heterogeneous graphs are never pooled as independent statistical replicates.

| Initializer | Mean signed-cut difference |
| --- | ---: |
| original paired K-MeansNM | 0.000 |
| independent | -0.649 |
| sample-antipodal | -0.686 |
| sample-pair | -0.536 |
| farthest-pair | -0.641 |
| spherical-pp | -0.615 |
| mean-perturb | -0.379 |
| PCA antipodal | -0.756 |
| GW screen-16 | +0.059 |

The differences are small relative to graph cut scales. The screen-16 marginal positive mean is descriptive, not compelling evidence of dominance: unequal work budgets, graph correlations and multiple comparisons remain. Within-graph paired bootstrap confidence intervals (4,000 deterministic bootstrap replicates) are in `paired_vs_original_kmeansnm.csv`; they do not prove an approximation ratio or allow naive independent graph-level significance claims.

## Exact verification

From the repository root on Windows PowerShell, using the recorded environment:

```powershell
python -m pytest tests/test_eva_full_gset_completion.py tests/test_eva_kmeans_init_all.py tests/test_eva_kmeans_init_full.py -q
python -m maxkcut.eva_full_gset_completion_cli inventory
python -m maxkcut.eva_full_gset_completion_cli aggregate
Get-Content "results/gset/experiment021-complete-71-gset/statistics/report.json"
```

Strict aggregation raises on **any** missing factor, incomplete graph, method/restart mismatch or total row count different from 31,950. The successful report contains `complete_graphs=71`, `no_EVA_factor_count=0`, `incomplete_candidate_count=0`, `runs_in_all_runs_csv=31950`, and `experiment021_status=all_71_complete`.

To independently recompute a missing EVA factor only if the verified archive is absent:

```powershell
python -m maxkcut.eva_cli --graphs G60 G61 G62 --backend numba --max-n 20000 --max-sweeps 10000 --tolerance 0.001 --seed 2026 --rounding-seed 2026 --restarts 20 --cpu-threads 1 --output-dir results/gset/experiment021-complete-71-gset/eva-numba
```

To reproduce the rounding, use the branch's `.github/workflows/experiment021-complete-gset.yml` command lines. Completed chunks are validated and reused. This workflow has a one-time path-limited push trigger for the user-requested initial launch and a manual dispatch path for controlled recovery; result commits do not trigger repeat calculations.

## Thesis–antithesis–synthesis

**Thesis:** Structured spherical clustering initialization can occasionally discover better local-search basins than the original antipodal K-MeansNM start.

**Antithesis:** Most of the alternative individual initialization means are slightly worse, and the portfolio gets nine families' opportunities instead of one.

**Synthesis:** Full canonical G-set coverage has been achieved as a reproducible **method-comparison dataset**, not as a proof of improved approximation guarantees. The next independent experiment should impose equal total computational budgets and evaluate normalized graph-level paired effects with appropriate multiplicity correction; do not overwrite Experiment021 results.

## Research handoff

- Branch: `experiment021-complete-71-gset`. PR: [#51](https://github.com/heri-espino/maxkcut/pull/51). Base: `experiment020-all-gset-paired-inits`.
- Runner: [successful Actions run #40831249367](https://github.com/heri-espino/maxkcut/actions/runs/40831249367), including test execution, per-factor convergence, full search, aggregation and publish.
- New implementation: `src/maxkcut/eva_full_gset_completion_cli.py`; tests: `tests/test_eva_full_gset_completion.py`.
- Original Experiment020 and Experiment019 factors, partitions and score traces remain the unmodified source of truth for 68 graphs.
- All 71 graph partitions, intermediate scores, provenance and statistical outputs are committed to GitHub; nothing depends on the runner's local scratch directory.
