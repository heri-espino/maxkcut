# Experiment030 — Weighted local-field clustering on frozen EVA embeddings

**Status: COMPLETE, independently verified and archived (2026-11-16 UTC).** The Windows experiment processed **all 1,000 archived EVA factors** and evaluated **6,000 hard Max-Cut partitions**, including exact SHA-matched reproduction of every Experiment029 baseline. There were **no new EVA or SDP solves**. The original compute [Actions #43965555603](https://github.com/heri-espino/maxkcut/actions/runs/43965555603) passed hosted synthetic tests and Windows numeric work, but its final hosted git publication failed on a single trailing blank Markdown line. The [hosted-only recovery #43966589165](https://github.com/heri-espino/maxkcut/actions/runs/43966589165) independently verified all **6,000 signed partitions**, all **104,242 clustering iteration records**, complete per-graph manifest checksums, and committed the archive **without rerunning clustering or EVA**.

## Results and interpretation

The primary raw field `-H` versus original `Y`, using 100 paired seeds **within each graph and dimension**, gave the following signed-cut mean differences and nominal 95% paired percentile bootstrap intervals (5,000 resamples):

| Graph | p=41: mean Δ [95% CI] | p=800: mean Δ [95% CI] | Best raw-field vs Y at p=41 | Best raw-field vs Y at p=800 |
| --- | --- | --- | --- | --- |
| G1 | −0.43 [−1.96, +0.94] | −1.41 [−3.56, +0.36] | 11,471 vs 11,471 | 11,474 vs 11,469 |
| G6 | −0.92 [−2.08, +0.19] | −1.35 [−3.39, +0.33] | 1,994 vs 2,001 | 1,994 vs 1,992 |
| G11 | +0.90 [+0.18, +1.68] | +0.90 [+0.04, +1.76] | 534 vs 534 | 536 vs 536 |
| G14 | −2.56 [−4.10, −0.96] | −2.92 [−4.43, −1.32] | 2,988 vs 2,985 | 2,988 vs 2,984 |
| G18 | +2.41 [+0.79, +4.03] | +1.64 [+0.09, +3.25] | 933 vs 927 | 933 vs 926 |

The equal-graph mean relative improvement was **+0.0602% at p=41** and **+0.0349% at p=800**. This differs from the unnormalized mean of raw cut-unit differences (−0.120 and −0.628) because graph cut scales differ. Gains at G11/G18 and losses at G14 are **heterogeneous**: there is no uniform improvement across five graphs. The CIs are conditional on the selected graphs, do not cover graph-population variation, and are **not corrected for comparing multiple methods/graphs**. Some best-of-100 cuts improve even where mean quality declines; best-of-100 is a secondary descriptive metric, not the primary evidence.

**Mechanism confirmed numerically.** For nonzero fields at a true EVA fixed point \(h_i=-d_i y_i\) with \(d_i=\|h_i\|\). The minimum observed cosine between \(y_i\) and \(-h_i/d_i\) across 1,000 factors was **0.999999611** (default width) and **0.999999636** (full width), with **zero null-field vertices**. Clustering normalized \(-H\) nearly reproduces the exact original baseline; raw \(-H\) and \(d_iY\) gave nearly identical graphwise means, as predicted by the fixed-point identity. Their different outcomes from the original \(Y\) arise predominantly from **vertex-specific weighting inside the centroid update**, not a novel angular embedding.

**Controls.** Graph G18's raw-field mean improves over the abs-degree-only representation by **+4.97 signed cut units at p=41** and **+4.03 at p=800** (exploratory comparisons), while the abs-degree-only mean itself is below the original baseline. This is evidence that weighted local-field magnitude may convey signal beyond absolute weighted degree for this fixed graph. Coherence-only \(q_iY\) does not reproduce raw-field gains there. Conversely, the raw field degrades G14's mean. No general algorithmic performance dominance or global SDP optimality follows.

**Complete archive:** [all 6,000 rounds](../../results/gset/experiment030-local-field-clustering/all_rounds.csv), [paired contrasts](../../results/gset/experiment030-local-field-clustering/paired_differences.csv), [paired/bootstrap table](../../results/gset/experiment030-local-field-clustering/per_graph_method.csv), [full scientific report](../../results/gset/experiment030-local-field-clustering/report.md), [deterministic normalized-effect figure](../../results/gset/experiment030-local-field-clustering/raw_field_relative_effects.svg), [case manifests, packed labels and full traces](../../results/gset/experiment030-local-field-clustering/graphs/). Figure regenerated with `python scripts/experiment030_plot_effects.py` and verified using `--check`. Binary EVA factor files remain in Experiments028/029 and are referenced by SHA, not recopied.


## Question and theory

For the weighted signed adjacency matrix \(W\), EVA returns a unit-row matrix \(Y\in \mathbb R^{n\times p}\) and its local field is \(H=WY\). At an exact coordinate fixed point with \(h_i\ne0\), \(h_i=-\|h_i\|y_i\). Thus using \(-H\) rather than \(Y\) does not add directional information; it changes row *magnitudes*, effectively weighting each vertex's contribution to spherical centroid updates. This suggests a useful controlled comparison that distinguishes local-field strength from pure degree and normalized coherence.

Let \(d_i=\|h_i\|_2\), \(a_i=\sum_j|w_{ij}|\), \(q_i=d_i/a_i\) when \(a_i>0\), and \(q_i=1\) for isolated vertices. By the triangle inequality \(0\le q_i\le1\). All inputs reuse **exactly the existing** antipodal K-MeansNM rounding routine, with its original `keep_best=True` signed Max-Cut objective. Per-method input representations:

| Method | Input rows | Purpose |
| --- | --- | --- |
| `baseline_y` | \(y_i\) | Frozen Experiment029 baseline, independently reproduced byte-for-byte |
| `unit_minus_h` | \(-h_i/d_i\), fallback \(y_i\) if \(d_i=0\) | Direction-only control; should approximate original \(Y\) when EVA converged |
| `raw_minus_h` | \(-h_i\), fallback \(y_i\) for zero fields | Raw field with vertex-dependent magnitude; antipodal initial partition is sign-invariant |
| `field_strength_y` | \(d_i y_i\) | Separates pure field-strength weighting from residual non-convergence angles |
| `coherence_y` | \(q_i y_i\) | Normalized signed-neighbor reinforcement, controls for weighted degree |
| `abs_degree_y` | \(a_i y_i\) | Weighted-degree-only control |

All positive row weights used by centroid updates are floored at \(10^{-12}\) for stable assignment to nonzero vectors. An isolated zero-field row retains its original \(y_i\) direction. All six methods are **postprocessing**: they never modify the saved EVA factors or experiment archives. The same seed draws the same antipodal initial K-MeansNM direction for all methods, graph, rank. No GW/Flip/KL/ILS refinement is used.

## Frozen data, design and hypotheses

- Original G-set: G1 random positive, G6 random signed, G11 toroidal signed, G14 planar-family positive, G18 planar-family signed, all \(n=800\). Source checksum from `results/gset/experiment022-graph-structure-methods/graph_features.csv` is validated (LF/CRLF-safe).
- From [Experiment028](experiment028-eva-fullrank-pilot.md): saved EVA factors for seeds 2026–2030; from [Experiment029](experiment029-eva-kmeansnm-100seed.md): saved factors for 2031–2125.
- Ranks: default \(p=41\), full \(p=800\). Exactly 100 EVA seed identities per graph/rank, **1,000 frozen factors**, six rounding methods each = **6,000 original signed cuts**, no new EVA/SDP solve.
- Primary comparison (preregistered): paired original signed-cut differences versus `baseline_y` *within each fixed graph, rank, seed*. 5,000 paired bootstrap resamples at each graph/rank/method; fixed seeded RNG 30030. Controls are `unit_minus_h` and `abs_degree_y`.
- Secondary: equality of `raw_minus_h` and `field_strength_y` cuts and the field alignment cosine between \(y_i\) and \(-h_i/d_i\). Numerical convergence tolerance is finite, so no exact alignment is assumed.
- Compare paired mean and best-of-100 cuts, wins/ties/losses, field norms/coherence, clustering iterations/time. Do not select a winner after inspecting all methods and claim independent held-out generalization. Five graphs limit between-graph inference. Nominal intervals do not adjust for comparisons.

## Execution and durability

New code: `scripts/experiment030_local_field_rounding.py`, synthetic tests `tests/test_experiment030_local_field_rounding.py`, workflow `.github/workflows/experiment030-local-field-clustering.yml`. The workflow gates execution on hosted tests and SHA checks, then uses only the dedicated self-hosted Windows X64 `PCAG103A-13` with an **isolated GitHub Actions checkout**, never the VM's active Experiment025 directory. Sparse `H=WY` is computed after selective Git LFS fetch of already-archived factors. No optimizer or Conda package installation on the VM. Max five graph workers, one BLAS thread each.

Results under `results/gset/experiment030-local-field-clustering/` only:

- `graphs/Gx/cases.csv`, `partitions.npz`, `trajectories.jsonl`, `manifest.json`: all 1,200 graph-specific cuts, packed exact partitions, complete clustering iterations and SHA-256 provenance, factor SHA references without factor duplication.
- `all_rounds.csv`: 6,000 original signed scores; `paired_differences.csv` same number of baseline contrasts; `per_graph_method.csv`: paired graph-rank-method mean differences, 95% intervals, best cuts and W/T/L.
- `summary.json`, `report.md`: validated completion status and interpretation.
- Each new archived cut is *independently* rescored on signed G-set weights by a separate GitHub-hosted job. All original 1,000 baseline partitions are required to reproduce their SHA-256 checksums and exact cut scores. Each iteration must have contiguous traces and monotonic best-so-far scores.
- All new outputs are small relative to the EVA factors, and must be committed to the Git repository. Large EVA factors remain in existing Experiment028–029 Git LFS paths and are never duplicated.
- The workflow is a one-time push scoped solely to its workflow file, so subsequent notes/source commits do not repeat computation. After success, remove automatic triggers, record run/commit IDs, and merge only with the user's explicit permission.

### Reproduce without EVA

```powershell
$env:PYTHONPATH = 'src;scripts'
python scripts/experiment030_local_field_rounding.py --validate-only
python scripts/experiment030_local_field_rounding.py --workers 5
python scripts/experiment030_local_field_rounding.py --verify-only
```

The first command validates metadata/graph hashes; the latter two require all original EVA LFS factor bodies downloaded. A hosted archive-only validation can use `--verify-only` after producing Experiment030 outputs, without needing any EVA factor binaries.