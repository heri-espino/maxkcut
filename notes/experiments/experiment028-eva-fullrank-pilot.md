# Experiment028 — Full-dimensional EVA embedding and spherical clustering pilot

**Research question.** Does changing EVA's embedding width from the paper's
\(p_0=\lceil\sqrt{2(n+1)}\rceil\) to the full \(p=n\) materially change the
continuous vector-relaxation objective and the original signed Max-Cut scores
obtained by the **same** spherical clustering/rounding implementations?

**Status:** Protocol prepared. Numeric conclusions require a successfully
completed, archived runner job and the generated `report.md`. Do not interpret
this document as an executed experiment.

## Five benchmark graphs: all `n=800`

| Graph | G-set family | Weights | Edges | Default `p` | Full `p` |
|---|---|---|---:|---:|---:|
| G1 | random | positive | 19,176 | 41 | 800 |
| G6 | random | signed | 19,176 | 41 | 800 |
| G11 | toroidal | signed | 1,600 | 41 | 800 |
| G14 | planar-family | positive | 4,694 | 41 | 800 |
| G18 | planar-family | signed | 4,694 | 41 | 800 |

The labels *planar-family* and *toroidal* describe the source generators; no
claim of graph-theoretic planarity is needed. Every graph is read from the
canonical `data/Gxx.txt` and verified against
`results/gset/experiment022-graph-structure-methods/graph_features.csv`.
**No G-set inputs may be modified.**

## Experimental design

- Five EVA initialization seeds **2026, 2027, 2028, 2029, 2030** for *each*
  graph and each rank: **5 graphs × 5 seeds × 2 ranks = 50 factor solves**.
- Use the unmodified **compiled sequential Numba EVA implementation**:
  `src/maxkcut/eva_numba.py::solve_eva_numba` and original
  `src/maxkcut/eva.py` objective; `float64`; angular convergence threshold
  `0.001` radians; at most 1,000 sweeps; checkpoints every 25 sweeps.
  Record and disclose nonconvergence; never equate stopping with SDP optimality.
- For *each* factor, use one fixed, independently seeded draw per method:
  Goemans–Williamson (`FJ_EVA`); spherical two-cluster Lloyd
  (`K-Means2N_EVA`); antipodally initialized spherical Lloyd with best
  intermediate cut retained (`K-MeansNM_EVA`). All use existing
  `src/maxkcut/rounding.py` unchanged, max 100 clustering iterations.
- RNG for a rounder is `SeedSequence([seed, graph_number, 2, method_index, 28])`.
  Corresponding rank conditions use the same seed identity and rounder label.
  **This is not identical starting geometry:** EVA produces vectors in
  different spaces, so the two underlying starting points are incomparable.
- Signed original-edge scores are computed with the repository's exact
  `edge_arrays` / `cut_weight`, never k-means inertia.
- Do **not** use Flip, KL, ILS or additional rank-dependent refinements in
  the primary analysis. This isolates the embedding + first rounding stage.
  Additional refined-cut comparisons can be pre-registered as Experiment029.
- Five independent graph-level processes on the 32-logical-processor
  university runner (not 50 concurrent workflows). Each is single-threaded
  for BLAS/Numba, has disjoint checkpoint paths, and never writes to a
  shared file during optimization. No existing EVA factors are modified.

## Interpretation and limits

The larger factor space contains embeddings of the smaller one, hence the
**global** rank-`n` optimum of the factor objective cannot be lower than the
global rank-41 optimum. EVA, however, is a *nonconvex* coordinate-ascent
heuristic: its **achieved** objective may be lower for `p=n`, due to different
optimization dynamics and stopping conditions. Maximizing the continuous
objective also does **not** order rounded cut scores in every instance.

For each graph and method compare matched-seed differences:

\[
\Delta_{g,s,m} = C_{g,s,m}(p=n)-C_{g,s,m}(p=41).
\]

Summarize means, best-of-five, win/tie/loss by graph, and EVA objective/runtime
tradeoffs. With only five seeds and five graphs, results are **exploratory**;
no broad claim of rank benefit or statistical significance is warranted.

## Reproduce on a dedicated, isolated checkout

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location).Path 'src')
python scripts/experiment028_fullrank_pilot.py --validate-inputs-only
python scripts/experiment028_fullrank_pilot.py --workers 5
python scripts/experiment028_fullrank_pilot.py --check-only
```

**Do not run these from the existing VM `C:\Users\175199\maxkcut`
checkout** or change its active branch. The scoped workflow
`.github/workflows/experiment028-fullrank-pilot.yml` uses a separate Actions
working directory and verifies `RUNNER_NAME=PCAG103A-13`, Windows PowerShell
5.1, and the installed `maxkcut` Conda environment. Never dispatch alongside
an active local VM study. No solver or dependency updates are permitted.

## Persistent outputs and recovery

- Output root: `results/gset/experiment028-eva-fullrank-pilot/` ONLY.
- `runs/Gxx/{default,full}/seed-NNNN/factor.npz`: complete EVA vectors and
  checkpoint RNG metadata for reproducibility / resume; large factors under
  new scoped Git LFS rules.
- `eva_sweeps.jsonl`: exact per-sweep continuous objective, angular changes,
  convergence flag, and cumulative time.
- `partitions.npz`: packed, independently rescored GW and clustering cuts.
- `case.json`: immutable settings, SHA-256 factor/partition hashes, graph
  provenance, and all per-method scores.
- `all_runs.csv`, `paired_differences.csv`, `summary.json`, `report.md`: small
  compact analyses, committed first; factor files subsequently committed via
  LFS if GitHub quota and Actions permissions permit. Action artifact upload
  retains a copy even if publication fails.
- Always inspect GitHub Actions artifact status and LFS success separately;
  an Actions artifact with finite retention is **not** a permanent off-VM
  archive. If LFS fails, publish a durable external archive with checksum
  verification instead of claiming source-of-truth persistence.
- If interrupted, resume only using checkpoint SHA/identity in this
  experiment directory; never overwrite historic results.

## Next steps

1. Confirm the hosted unit/synthetic-data preflight passed.
2. Confirm five-graph/50-run coverage and **150 independent cut scores**.
3. Audit every original signed cut and per-factor unit-vector constraint.
4. Compare `report.md` and the `paired_differences.csv` by graph and method.
5. Preserve the complete factor/iteration archive in Git LFS, and keep this
   branch isolated until the user explicitly authorizes merging.