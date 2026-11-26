# Experiment032 — Compare GW and K-MeansNM on EVA embeddings versus signed local fields across all 71 G-set graphs

**Status: COMPLETE, independently audited and archived (2026-11-16, Mexico City).** This is a new, independent study following completed Experiments030 (five-graph field-weighted K-MeansNM) and 031 (five-graph GW/sign geometry), which remain in separate unmerged PRs #87/#88. Experiments028 and 029 are merged into `main`. Do not alter or rerun their factors.


## Verified findings (71 graphs, 50 paired starts each)

The three-stage [Actions run #44012539461](https://github.com/heri-espino/maxkcut/actions/runs/44012539461) **passed hosted six-test preflight, Windows PCAG103A-13 computation, and an independent hosted SHA+original-signed-cut audit of all 14,200 partitions**. Every one of the 71 graph manifests, 200 exact packed partitions per graph, and full K-MeansNM Lloyd best-so-far histories was archived. All 71 original graph/legacy EVA SHA source identities matched; **71 reused factors, zero EVA/SDP solves**. The sole initial preflight failure was a missing hosted `threadpoolctl` test dependency, corrected before the successful compute. No outputs from the failed preflight were used.

**Primary endpoint:** paired representation effect (raw `-H` minus `Y`) in original signed Max-Cut cut weight; first average over the 50 matched rounding normals **within each graph**, scale by the corresponding `Y` mean cut, then average **equally over graphs**.

| Rounding method | All 71: mean relative change | Positive-only 35 | Signed 36 |
|---|---:|---:|---:|
| GW single hyperplane | **+0.000198%** | +0.000156% | +0.000239% |
| Antipodal K-MeansNM, best cut along up to 100 Lloyd iterations | **+0.023957%** | +0.001531% | **+0.045759%** |

K-MeansNM `-H` versus `Y` achieved higher **50-start mean cut on 42 graphs, tied on 3 and worsened on 26**. The 71-graph mean raw signed cut difference was **+0.721 units**, but heterogeneous graph sizes/weights make the graph-normalized percentage statistic more informative. The mean fraction of GW vertex-hyperplane classifications that differ between `Y` and `-H` was **0.006459%**; the minimum saved cosine between unit `Y` and normalized `-H` over all 71 graphs was **0.999999510**, demonstrating approximate EVA fixed-point invariance.

### Graph-structure heterogeneity (exploratory)

| Family | Positive-only mean change (n) | Signed mean change (n) |
|---|---:|---:|
| Random | −0.004303% (18) | **−0.047400%** (12) |
| Planar-family | +0.009360% (14) | **+0.138600%** (10) |
| Toroidal | 0.000000% (3) | **+0.059294%** (14) |

The largest observed normalized mean increases were G19 (+0.4463%), G11 (+0.4101%), G42 (+0.3618%) and G18 (+0.3246%). Declines included G8 (−0.3581%), G27 (−0.2359%) and G40 (−0.2057%). These illustrative cases must not be treated as independent validation or used to select a supposedly universal best method. Signed positive-only status is confounded with graph geometry; the study **does not prove that negative edge weights cause the signed-family differences**.

**Geometric mechanism:** for exact nonzero-field EVA coordinate optima, `-H=DY`, so GW signs are invariant to positive vertex-specific row scaling. K-MeansNM centroid updates depend on magnitudes `D_ii=||h_i||`, so its output can change, positively or negatively. The 50 matched restarts are **rounding randomness conditional on one frozen EVA solution per graph**; they are not 50 independent EVA optimizations. Graphwise 95% percentile CIs are unadjusted for multiple comparisons, G-set graphs/families are correlated, and no signed-graph GW approximation guarantee or SDP global optimality follows.

**Durable outputs:** [full all-71 report](../../results/gset/experiment032-all-gset-field-rounding/report.md), [14,200 original signed scores](../../results/gset/experiment032-all-gset-field-rounding/all_runs.csv), [71 graphwise paired/bootstrap summaries](../../results/gset/experiment032-all-gset-field-rounding/per_graph.csv), [coverage audit](../../results/gset/experiment032-all-gset-field-rounding/summary.json), and [71 independent per-graph packed partitions, SHA manifests and Lloyd histories](../../results/gset/experiment032-all-gset-field-rounding/graphs/). The updated Marp deck appends seven new slides with two source-derived SVG plots; the generator `scripts/experiment032_plot_field_rounding.py` is fully deterministic and requires no optimization.

## Formal design and research question

For each canonical undirected weighted G-set graph \(G=(V,E,W)\) and its **one originally archived, SHA256-verified, internally converged EVA factor** \(Y\in\mathbb R^{n\times p}\), rows \(\|y_i\|=1\), define \(H=WY\) and \(Z=-H\) (with fallback \(z_i=y_i\) if \(\|h_i\|\le10^{-12}\)). All archived factors use the prior automatic rank \(p=\lceil\sqrt{2(n+1)}\rceil\). This is **not** another 100-EVA-initialization study; here, each graph contributes precisely one historically frozen factor, and the 50 restarts quantify *rounding* randomness conditional on that factor.

- Dataset: **all 71 canonical G-set graphs** (`data/G*.txt`): G1–G67, G70, G72, G77, G81. The authoritative 71-entry [Experiment021 factor inventory](../../results/gset/experiment021-complete-71-gset/inventory.json) identifies 59 factors in Experiment002, nine in Experiment003, and three in Experiment021. Every factor SHA256 and its original EVA case/graph identity are verified before use. All G-set weights, including negative weights, are preserved.
- Matched initialization: **50 restarts** \(t=0,\ldots,49\) per graph, with one Gaussian normal \(r_{g,t}\) from `SeedSequence([2026, g, 2, t, 0])`, the original Experiment019/020 antipodal-seed convention. The **same normal vector** initializes GW and K-MeansNM for both \(Y\) and \(-WY\). Independent normals across distinct graphs and dimensions are not equivalent geometrically.
- Four arms, no local refinement: `GW_Y`, `GW_minusH`, `KM_Y`, `KM_minusH`. Each GW arm takes exactly one hyperplane cut. Each K-MeansNM arm starts with \(r\) and \(-r\), runs up to 100 spherical Lloyd iterations and keeps the **best original signed graph cut seen along the trajectory**. The initial K-Means cut equals its GW counterpart up to global complement. **No Flip, KL, ILS, Tabu, or extra random restarts** are applied.
- Total new rounding: \(71\times50\times4=\mathbf{14,200}\) exact signed feasible cut records, of which 7,100 are GW arms and 7,100 are K-MeansNM arms, forming **7,100 paired within-algorithm representation contrasts**. The 71 EVA factors are reused, with **zero EVA/SDP solves**.

The signed Max-Cut objective is \(C(x)=\sum_{\{i,j\}\in E}w_{ij}\mathbb1\{x_i\ne x_j\}\). For each graph, algorithm and restarter, the paired contrast is
\[
\Delta_{g,t}^{(A)}=C\big(A(Z;r_{g,t})\big)-C\big(A(Y;r_{g,t})\big),\quad A\in\{\mathrm{GW},\mathrm{KM}\}.
\]
**Raw `-H` and `H` are GW-cut equivalent** at an exact EVA fixed point since they globally invert labels; `-H` is chosen so its direction aligns with `Y` for more interpretable matched signs. In contrast, for K-MeansNM at convergence, \(-H\approx DY\), \(D_{ii}=\|h_i\|\), changes each vertex's weight in centroid updates. The two outcomes should therefore differ much more for KM than GW.

## Fairness, independence and limits

The same input factor, graph, 50 matched normals, Lloyd procedure and *original signed scoring objective* are held fixed within each pair. The first K-Means assignment uses the corresponding GW cut; hence K-Means with `keep_best=True` must **never score worse** than its own GW arm on the same direction. Shared random normals do not imply equal runtime; K-Means can run different iteration counts and field computation adds preprocessing cost, reported separately if material.

For each graph: report paired mean score difference, mean relative percentage (\(100\bar\Delta/|\overline{C(Y)}|\)), 95% percentile paired bootstrap CI over its 50 **rounding** restarts (3,000 fixed-seed resamples), wins/ties/losses, best-of-50, signed-only and positive-only subsets, and mean GW sign mismatch. Across graphs aggregate percentages with **one vote per graph**, not by pooling raw signed scores or treating 50 restarts as 50 new graphs. The 71 instances are structurally related; across-graph inference must disclose dependency and avoid unjustified population-level statistical significance. Underlying EVA factors are fixed, so one cannot infer across EVA initializations from these data.

## Execution and archive invariants

- Code: `scripts/experiment032_all_gset_field_rounding.py`, with six small synthetic regression tests in `tests/test_experiment032_all_gset_field_rounding.py`. The driver cross-checks the first two K-Means restarts per graph against the unchanged repository `round_spherical_kmeans`, including iteration count and packed labels. It preserves per-iteration original signed cut and best-so-far traces.
- A hosted validation gate checks all 71 graph and EVA case metadata and exact source graph SHA256 (including faithful LF/CRLF equivalents). Then the dedicated self-hosted Windows x64 **PCAG103A-13** executes on a separate Actions checkout, with the already installed `maxkcut` Conda environment, up to five graph workers and one BLAS thread per worker. **Do not touch the VM's active Experiment025 project worktree.** No Git LFS materialization is needed for the historical Experiment002/003/021 factors; no Experiment028/029 full-rank factors are fetched.
- New output folder only: `results/gset/experiment032-all-gset-field-rounding/`. For each graph, `graphs/G*/cases.csv`, `partitions.npz` (200 exact bit-packed hard cuts), `traces.jsonl` (complete Lloyd best-seen history), `manifest.json` with SHA256 of every file and the original source factor/graph. The graph writer refuses partial prior output and independently rescales every saved partition before returning.
- Combined outputs: `all_runs.csv` (14,200 score rows), `per_graph.csv` (71 rows), `summary.json` with strict coverage, and a scientifically caveated `report.md`.
- An **independent hosted audit** reconstructs all 14,200 cuts directly from original signed G-set edge lists and packed partitions, checks all 71 SHA manifests, and commits outputs without accessing EVA factor bytes. If publication fails after computation, recover the uploaded Actions artifacts; **never rerun finished numerical jobs just to fix Git formatting or reporting**.
- The verified numerical result was published without EVA reruns. The one-shot workflow must be switched to **manual-only** after success; PR #89 stays unmerged until the user authorizes integration. The original seven-slide Experiment029 addendum remains untouched, and seven new Experiment032 result slides have been appended to form a **111-slide** canonical Marp deck, with two deterministic SVGs and strict 111-page PDF/HTML export validation.

## Reproduction

From a clean repository checkout with source factors present:

```bash
PYTHONPATH=src:scripts python -m pytest -q tests/test_experiment032_all_gset_field_rounding.py
PYTHONPATH=src:scripts python scripts/experiment032_all_gset_field_rounding.py --phase validate
PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/experiment032_all_gset_field_rounding.py --phase compute --workers 5
PYTHONPATH=src:scripts python scripts/experiment032_all_gset_field_rounding.py --phase verify
PYTHONPATH=src:scripts python scripts/experiment032_all_gset_field_rounding.py --phase aggregate
```

**Do not run the last three commands if the archived Experiment032 outputs already verify.** All committed source factors and historical Experiment025 numerical outputs are immutable research artifacts.