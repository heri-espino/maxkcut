# Experiment020 — Final G-set paired initializer comparison (verified 2026-10-29)

## Scope, provenance and completion

Research question: Does changing the **two-centroid spherical K-MeansNM initialization**, while holding the saved EVA factor and the signed Max-Cut refinement pipeline fixed, improve cut quality robustly across graphs?

**Status: complete for every graph with a completed hash-verified EVA factor.** GitHub Actions recovery run [40826652014](https://github.com/heri-espino/maxkcut/actions/runs/40826652014) finished successfully. Previous run [40823065904](https://github.com/heri-espino/maxkcut/actions/runs/40823065904) failed because Windows workers raced while publishing JSON summaries; the repaired code generates per-graph summaries only after all workers have joined, and resumes only incomplete graphs.

- Canonical scope: 71 graphs from `data/G*.txt`; 68 eligible saved EVA factors; **67 newly completed graphs**, plus **G63 reusing verified Experiment019**.
- **G60, G61, G62 excluded** because no completed matching original EVA factors were found. They were neither silently counted nor recomputed.
- Nine initializers × 50 matched restart IDs per graph = **450 runs/graph, 30,600 saved runs**.
- Completed graph folders: `results/gset/experiment020-all-gset-paired-inits/G*/`; reused G63: `results/gset/experiment019-kmeans-init-full-pipeline/`.
- Cross-graph outputs: `results/gset/experiment020-all-gset-paired-inits/statistics/{report.json,coverage.csv,all_runs.csv,per_instance_methods.csv,paired_vs_original_kmeansnm.csv}`.
- `report.json`: `complete_graphs=68`, `completed_new_graphs=67`, `reused_existing_graphs=1`, `no_EVA_factor_count=3`, `incomplete_candidate_count=0`, `runs_in_all_runs_csv=30600`.
- Exact graph/factor SHA256 fingerprints, original signed graph edges, all stages' bit-packed binary partitions, detailed trial traces, seeds, and case fingerprints are retained. Never confuse completed cut searches with newly solved or certified convex SDPs.

## Frozen protocol

**Initializer names:** `paired` (original antipodal K-MeansNM control), `independent`, `sample-antipodal`, `sample-pair`, `farthest-pair`, `spherical-pp`, `mean-perturb`, `pca-antipodal`, `gw-screen16`.

Fixed per-restart pipeline: spherical Lloyd k=2 (up to 100 iterations, retain best actual signed cut along trajectory) → positive flips (20 sweeps) → generalized KL (30 passes) → 80-cycle ILS (8 inner KL passes, kicks 8/16/32/64, temperature 12). Same saved EVA factor **within each graph**; no EVA refit and no graph transformation.

RNG protocol (graph number `g`, restart `r=0,...,49`):
- Raw initialization: `SeedSequence([2026,g,2,r,0])` for paired control; stream `1` for alternatives.
- Flip: `[2026,g,r,1313]`; KL: `[2026,g,r,101]`; ILS: `[2026,g,r,1401]`.
- Thus each alternative is *matched on restart index and postprocessing seed* with the control; different initializers are **not** identical raw random states.
- `gw-screen16` spends 16 additional initial signed cut evaluations per restart (54,400 across 68 graphs); PCA has preprocessing cost. Comparison of cut quality should disclose the unequal candidate evaluation budget.

## Verified numerical results

From the committed `per_instance_methods.csv` and `paired_vs_original_kmeansnm.csv`, using **each graph as one unit**. `mean difference` is averaged over the 68 graphwise matched mean-cut differences (method minus paired control), in cut-weight units; it is **not** the mean of raw cut scores pooled across graphs.

| Initializer | Mean difference (cut units) | Graphs with higher mean | Graphs with lower mean | Graphs tied for best observed cut (of 68) |
|---|---:|---:|---:|---:|
| paired (control) | 0 | — | — | 13 |
| independent | -0.655 | 20 | 44 | 16 |
| sample-antipodal | -0.602 | 19 | 45 | 18 |
| sample-pair | -0.469 | 26 | 39 | 14 |
| farthest-pair | -0.594 | 27 | 37 | 18 |
| spherical-pp | -0.586 | 25 | 40 | 18 |
| mean-perturb | -0.327 | 25 | 40 | 16 |
| pca-antipodal | -0.797 | 26 | 39 | 16 |
| gw-screen16 | +0.119 | 34 | 31 | 24 |

Remaining graphs for each method had exactly tied average cuts; best-cut ties across initializers count for each tied initializer. These are descriptive statistics; method selection and multiple comparisons require care.

**Best-of-nine improves on the control's best-of-50 on 55/68 graphs; it ties on 13/68.** This is a **portfolio/selection effect**, not evidence that a single initializer dominates. The nine-way search uses nine times as many initialization families and cannot be compared to one method's 50 trials as if it had equal computational cost.

**G63:** farthest-pair reaches 26,906 (Experiment019 reused); paired original reaches 26,897 in this run. Historical best feasible G63 cut remains **26,917**, and the exploratory **27,047 target is not a certified optimum**. The best Experiment020 cut is 11 below the historical result, 141 below the target.

**G11:** the independent initializer reaches a feasible **564** on the canonical graph, matching the independently established G11 optimum of 564 recorded in `docs/gset-reference-values.md` (subject to the saved canonical graph identity). This is an achieved cut value, not a new proof.

**Additional best-of-nine gains vs paired best-of-50:** G59 +28 (gw-screen16); G41 +18 (independent); G64 +15 (mean-perturb); G66 +12 (farthest-pair); G81 +12 (gw-screen16). Absolute differences across graph sizes should not be pooled as a scientific effect size.

### Independent graph-level sensitivity analysis

For each graph `g` and method `m`, define `d[g,m] = 100*(mean_cut[g,m]-mean_cut[g,paired])/abs(mean_cut[g,paired])` (percentage points). The unweighted mean over 68 graph-level percentages for `gw-screen16` is approximately **-0.0032%**, with an illustrative 5,000-replicate graph-resampling percentile interval **[-0.0197%, 0.0137%]** (seeded bootstrap). This interval includes zero. The slight positive unnormalized mean difference (+0.119 cut units) is not robust evidence of practical domination.

The committed 4,000-replicate paired bootstrap intervals in `paired_vs_original_kmeansnm.csv` are **within each graph over its 50 restart IDs**. These are not CIs for graph-optimality and cannot substitute for across-graph inference. Treat correlated G-set families and simultaneous method comparisons as additional limitations.

## Important historical-target metadata caveat

The Experiment019 CLI was reused for Experiment020 with **G63-only labels** `historical_best=26917`, `target=27047`, `delta_vs_26917`, and `gap_to_target` in the **per-graph** `leaderboard.json`/`summary.csv` for graphs other than G63. Those numerical **gap fields are invalid as graph-specific comparisons**; ignore them for G1–G59, G64–G67, G70, G72, G77, G81. Use actual `raw_cut`, `flip_cut`, `kl_cut`, `final_cut`, and graphwise comparisons from `statistics/` instead. **Do not imply a score of 26,917 is a historical reference for any graph except G63.**

The per-instance statistics aggregator does **not** use those G63-only gap fields. The maximum observed final cut across all 68 analyzed instances is 26,906 (G63), below 27,047, so the inappropriate shared target did not prematurely stop an ILS search on the recorded runs. A follow-up *metadata-only repair* should remove G63-only fields from other graphs' leaderboards; no EVA or local search recomputation is warranted.

## Interpretation and next research experiment

**Thesis:** initialization diversity yields occasional superior cuts; the nine-family portfolio improves the best observed result on 55 graphs.  
**Antithesis:** most individual alternatives have slightly lower mean paired cut than original K-MeansNM; screening uses extra evaluations; no new G63 record emerged.  
**Synthesis:** preserve the full 30,600-run dataset as a completed method-comparison study. Subsequent work should compare a **fixed total computation budget** for the nine-initializer portfolio versus additional paired K-MeansNM restarts, with graph-level normalized paired effects and multiple-comparison controls. A genuinely new experiment would be Experiment021 and must not be folded into Experiment020 after the fact.

## Reproducibility / handoff

Workflow: `.github/workflows/experiment020-self-hosted.yml`. CLI: `python -m maxkcut.eva_kmeans_init_all_cli aggregate --output-root results/gset/experiment020-all-gset-paired-inits --restarts 50 --require-complete`.

Small regression test: `python -m pytest tests/test_eva_kmeans_init_all.py tests/test_eva_kmeans_init_full.py -q`.

Original full processing command (only needed for a missing verified graph; **do not** repeat finished searches): `python -m maxkcut.eva_kmeans_init_full_cli --graphs G2 --factor-dir results/gset/experiment002-eva --output-dir results/gset/experiment020-all-gset-paired-inits/G2 --methods paired independent sample-antipodal sample-pair farthest-pair spherical-pp mean-perturb pca-antipodal gw-screen16 --restarts 50 --chunk-size 5 --workers 32 --kmeans-iters 100 --flip-sweeps 20 --kl-passes 30 --ils-cycles 80 --ils-kl-passes 8 --kick-sizes 8 16 32 64 --temperature 12 --target 27047`. Only graph G63 has a meaningful 27,047 benchmark target; the shared flag on historical Experiment020 runs is recorded solely for faithful reproducibility.

A following agent should read **AGENTS.md**, this note, Experiment019's note, the two CLI modules and tests before editing or claiming numerical advances. No PR merge is authorized by this record.
