# Experiment031 — Signed-edge geometry and GW random-hyperplane diagnostics

**Status: protocol and code staged.** Experiment030 (field-weighted K-MeansNM) is completed in unmerged PR #87. Experiment031 is a *different*, diagnostic-only continuation, using factors already merged from Experiments028/029; it makes no claims about data until the new workflow completes.

## Question and mathematical frame

The signed Max-Cut objective is \(C(x)=\sum_{\{i,j\}\in E}w_{ij}\mathbf1\{x_i\ne x_j\}\), with original signed weights. For saved EVA unit vectors \(y_i\), define \(\theta_{ij}=\arccos(y_i^\top y_j)\). With a Gaussian GW normal \(r\sim\mathcal N(0,I_p)\):

\[
\Pr[\mathrm{sign}(r^\top y_i)\ne\mathrm{sign}(r^\top y_j)]=\theta_{ij}/\pi,\qquad
\mathbb E[C_{\rm GW}(Y)]=\sum_{\{i,j\}\in E}w_{ij}\theta_{ij}/\pi.
\]

The **positive** part of the expected cut is nonnegative and rewards large angles, while the **negative** part is nonpositive and penalizes large angles. This is an edgewise preference, not a guarantee of local satisfaction in a frustrated graph. A negative edge does not automatically send one endpoint across the hyperplane; the preferred geometry generally aligns its two endpoints. The angular identity holds regardless of the sign of \(w_{ij}\); it is the weighted objective that changes sign.

For the original signed adjacency \(W=W_++W_-\), with \(W_+=\max(W,0)\) and \(W_-=\min(W,0)\), define signed local fields \(H=WY=H_++H_-\). At a nonzero-field EVA coordinate fixed point \(-h_i=\|h_i\|y_i\), so matched random hyperplanes classify \(Y\) and \(-H\) identically (except numerical boundary/tie effects). We measure finite-tolerance deviations and also the projection decomposition

\[
r^\top h_i=r^\top h_{i,+}+r^\top h_{i,-}.
\]

A diagnostic **positive-only-field ablation** applies the same normal to \(-H_+\), *holding the original EVA factor Y fixed*. This is **not** the cut from optimizing a positive-only graph: the ablation isolates how negative neighbors contribute to the projected local field, not an alternative solution. For vertices with no positive field, it falls back to the original \(Y\) sign.

## Preregistered design, scope, and controls

- Five archived 800-node G-set graphs **G1, G6, G11, G14, G18**, including positive-only controls **G1/G14** and signed cases **G6/G11/G18**. Validate the exact original graph SHA-256 against Experiment022 metadata, accepting only normal LF/CRLF checkout conversions.
- Reuse **exactly 1,000 immutable EVA factor matrices** from Experiments028/029: 100 distinct EVA seeds **2026–2125** per graph and rank, at \(p=41\) and \(p=800\). Before reading each factor, compare its exact binary SHA-256 to the Experiment029 archived index and validate its original signed relaxation objective. Do **not** solve EVA/SDP, regenerate a factor, or modify its source.
- **16 reproducible Gaussian GW normals per factor**, generated from `SeedSequence([31031, graph_number, rank_index, eva_seed])`, within each factor's coordinate system. Apply exactly the same 16 normals to \(Y\), \(-H\), and \(-H_+\). This yields **16,000 matched normals and 48,000 independently auditable feasible signed Max-Cut scores**.
- Each factor gets: positive/negative edge-count and absolute weight, absolute-weight-weighted mean angle and cut probability, weighted fraction of geometrically unfavorable edges, angular-theory expected cut contributions for each sign, continuous relaxed objective, local-field cosine and norm summaries. Each normal gets three exact signed cut scores, finite-convergence label mismatch \(Y\) vs \(-H\), vertex changes due to dropping negative-field contributions, percentage of vertices with larger absolute negative than positive projected contribution, and signed-positive/signed-negative cut decomposition.
- Bootstrap the 100 **EVA-seed-level** means of the 16 paired \((-H)-Y\) signed cut differences (5,000 bootstrap replicates, fixed independent seed per graph/rank). Do **not** treat 16 directions on one factor as 16 independent graph instances. Across-graph extrapolation and simultaneous inference for the 5 graphs are not certified.
- Rank p=41 and p=800 have different vector spaces: identical seeds alone do not make their normals a geometrically matched pair. Only the three representations of the **same** factor share literal random normal coordinates.
- GW is only a one-shot sign assignment per hyperplane. **No K-MeansNM, Flip, KL, ILS, or post-rounding refinement** is run; there is no best-of-directions selection for the primary results.

## Code and reproducibility

- Driver: `scripts/experiment031_signed_gw_geometry.py`; synthetic tests: `tests/test_experiment031_signed_gw_geometry.py`.
- Source index: `results/gset/experiment029-eva-kmeansnm-100seed/all_runs.csv`. Factors referenced but never copied into results. `scripts/experiment029_kmeansnm_100seeds.py` provides the source graph and metadata audit.
- New standalone output directory: `results/gset/experiment031-signed-gw-geometry/`. Each `graphs/Gx/` contains source-linked `cases.csv`, 16-direction per-factor `trials.csv`, three independently rescored bit-packed partitions per normal in `partitions.npz`, and SHA-256 `manifest.json`. Global `all_cases.csv`, `all_gw_trials.csv`, `per_graph_rank.csv`, `summary.json`, and `report.md` are compiled from those five verified directories.
- Hosted tests and five original graph hashes must pass before dispatch to dedicated **PCAG103A-13** (self-hosted Windows X64, PowerShell 5.1) in an isolated GitHub Actions worktree; only existing Conda environment, no pip installs or solver runs on VM. Process at most five graph workers with one BLAS thread each. Fetch only frozen EVA *factors* through scoped Git LFS, not unrelated large results. Do not touch VM's active Experiment025 directory, raw data, or any solver implementation.
- Hosted independent publisher must verify all 48,000 archived original signed hard cuts and all graph/source/factor identities, without downloading or resolving any EVA factor, before committing compact outputs to the branch. Keep every source, tests, generated values, and provenance in the repo; no automatic merge.

Reproduce on an isolated checkout with the complete LFS factor files and installed `maxkcut` Conda environment:

```powershell
$env:PYTHONPATH = 'src'
python -m pytest tests/test_experiment031_signed_gw_geometry.py -q
python scripts/experiment031_signed_gw_geometry.py --validate-only
python scripts/experiment031_signed_gw_geometry.py --workers 5
python scripts/experiment031_signed_gw_geometry.py --check-only
```

## Interpretation rules

1. Compare sign-conditioned angle distributions **on original edges**. The average separation probability \(\theta/\pi\) is an *expectation over random normals*, not a deterministic per-edge cut label.
2. If \(-H\) and \(Y\) agree under GW, that is predicted by stationarity and provides an important null/control result, **not** a failed experiment.
3. Differences between \(-H\) and positive-only local fields measure the influence of negative neighbors on local projected fields **at fixed Y** and should not be marketed as performance improvement from changing the original graph.
4. Compare analytic and Monte Carlo signed expectation as a diagnostic for GW/edge-score implementation consistency. With 16 directions per case, stochastic error does not vanish; report uncertainties and conditional scope accurately.
5. The study does not prove a global SDP optimum, a Max-Cut approximation guarantee for signed weights, or a new universally superior rounding method.