# Experiment019 — Paired 50-start comparison: spherical KMeans initialization + Flips + KL + ILS

## Precise objective and historical reference

G63: original signed weighted graph, 7,000 vertices, fixed completed
EVA factor Y with 119 columns. No new EVA/SDP, no PCA projection
of the factor, no graph transformation, and no use of negative-weight
magnitude as if all edges were positive. The user reports the
best independently evaluated feasible cut as **26,917**, with
reference target **27,047** (not certified optimal).

Experiment018's small 3-restart raw-KMeans+KL initialization
pilot had best **26,867 (mean-perturb)**. Its raw+KL scores are
NOT directly comparable to the 26,917 full-pipeline best because
the latter included Flips and 80 cycles of iterated local search.

Therefore run all nine original Experiment018 k=2 initializers
on the **SAME 50 restart IDs (0–49)**, and evaluate the SAME
postprocessing after every k=2 spherical Lloyd clustering:

    Round → Flips → KL → ILS-KL

This experiment uses the exact best previous grid setting for
`SavedKMeansFlip+KL+ILS` in Experiment013
(config ID `503d49b5c2fa621f`):

- Flips: legacy archived KMeansNM-Flip initial source used
  20 positive-gain sweeps. Here apply the same
  `improve_by_flips` stage directly after each new initializer.
- Generalized indexed-heap KL: **30 passes**.
- ILS-KL: **80 cycles**, inner KL **8 passes**.
- ILS kick sizes: **(8, 16, 32, 64)**.
- ILS annealing temperature: **12**.
- Lloyd spherical k=2 iterations: **100**, retain highest
  signed cut observed during Lloyd.
- Existing project seed schedule: kmeans Gaussian
  `SeedSequence([2026, graph_number, 2, restart, stream])`,
  where paired uses stream 0 exactly as old KMeansNM and
  remaining initializers use stream 1 exactly as Experiment018.
  The postprocessing seeds are SHARED across methods
  (paired restart design, not independent postprocessor luck):
  Flip: `[2026, graph_number, restart, 1313]`;
  KL: `[2026, graph_number, restart, 101]`;
  ILS: `[2026, graph_number, restart, 1401]`.
  This is the same Flip/KL/ILS pipeline implementation and seed
  convention as `eva_pipeline_grid.evaluate_pipeline`.

Nine k=2 initializers: paired, independent, sample-antipodal,
sample-pair, farthest-pair, spherical-pp, mean-perturb,
pca-antipodal, gw-screen16. The latter explicitly spends
16 additional original-graph cut evaluations per start.
PCA computes its leading eigenvector with additional cost,
so compare mean/median wall time in addition to raw and
final Max-Cut.

## GitHub Windows self-hosted run

Workflow: `.github/workflows/experiment019-self-hosted.yml`.
Uses registered Windows runner `PCAG103A-13`, existing
Conda `maxkcut` environment and its Windows DLL paths.
Trigger only on a push to this dedicated experiment019
branch containing a change to the workflow file, or manually
via GitHub Actions after it is registered.

Test checks:
- Independently verified original G63 graph/factor files.
- Exact four-stage partition and scores parity with
  original Experiment013 `evaluate_pipeline` when k=2 paired.
- 9-method small synthetic Windows spawn ProcessPoolExecutor run.
- All cut scores on original signed weights and monotonicity:
  Round <= Flip <= KL <= Final.
- Distinct budget fingerprints, cache re-verification, intact
  original EVA SHA256.

**Full 50-start run**: nine methods x 50 restart IDs =
450 complete pipelines, chunk size 5, four workers,
up to 80 ILS cycles per start. No speculative score claims.

### Manual PowerShell from the repository root

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path "src"
python -m pytest tests/test_eva_kmeans_init_full.py -q

python -m maxkcut.eva_kmeans_init_full_cli `
  --graphs G63 `
  --restarts 50 `
  --chunk-size 5 `
  --workers 4 `
  --kmeans-iters 100 `
  --flip-sweeps 20 `
  --kl-passes 30 `
  --ils-cycles 80 `
  --ils-kl-passes 8 `
  --kick-sizes 8 16 32 64 `
  --temperature 12 `
  --target 27047
```

Completed outputs are stored in
`results/gset/experiment019-kmeans-init-full-pipeline/`.

- `summary.csv`: all initializer × restart rows with
  original signed graph **raw, flip, KL, final** scores,
  durations and exact (not estimated) gap to 27,047.
- `leaderboard.json`: best/mean/median final cut,
  best raw/flip/KL per method, mean CPU wall time,
  difference from historical 26,917, restart coverage.
- `cases/*.json`: exact seeds, SHA256 original source
  fingerprints, config identity, all optimization traces.
- `trials/*.jsonl`: detailed per-restart Lloyd/flip/
  generalized KL/80-cycle ILS diagnostics.
- `archives/**`: independently verified compact bit-packed
  binary Round / Flip / KL / Final partitions and original
  signed weighted edge list for every chunk.
- `factor-snapshots`: SHA-verified immutable original
  G63 EVA factor.
- `errors`: failure details if any; previously completed
  chunks remain auditable/resumable.

The workflow always uploads artifacts, and if successful
auto-commits results ONLY to the experiment019 branch.
It is safe to rerun the SAME command: matching completed
chunks are verified and skipped. Changing budgets produces
different hashed case identities. Extending to 100 starts
with the same chunk size/budgets reuses the first 50 starts.

**Statistical reading**: Compare distributions across matched
restart IDs, per-start paired differences vs KMeansNM control,
best, median, mean, cost and any success above 26,917.
A single best-score improvement is evidence of a better feasible
cut, but not proof that an initializer dominates in expectation
or solves Max-Cut globally.
