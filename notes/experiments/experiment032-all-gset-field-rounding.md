# Experiment032 — Compare GW and K-MeansNM on EVA embeddings versus signed local fields across all 71 G-set graphs

**Status: staged; no numerical performance claim until the 71-graph independent audit succeeds.** This is a new, independent study following completed Experiments030 (five-graph field-weighted K-MeansNM) and 031 (five-graph GW/sign geometry), which remain in separate unmerged PRs #87/#88. Experiments028 and 029 are merged into `main`. Do not alter or rerun their factors.

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
- A one-shot Actions workflow should be switched to **manual-only after success**, and the PR kept unmerged until the user authorizes integration. The seven-slide Experiment029 addendum to the 104-slide Marp presentation should remain untouched until **after** this Experiment032 run passes numerical/audit verification; then add a new reproducible, sourced final section with results, sign-stratified statistics, a derived SVG and all-71 graph coverage.

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