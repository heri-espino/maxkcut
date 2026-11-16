# Experiment029 — Antipodal K-MeansNM rank comparison with 100 EVA seeds

**Status:** Pre-registered, execution pending. Do not report numerical results until the
numerical run, 1000 signed partitions, LFS object preservation, and independent archive
checks complete. No existing experiment is modified or recomputed.

## Research question

Holding the five G-set graphs and **antipodal K-MeansNM** algorithm fixed, does
maximizing the EVA factor dimension from the paper's default
\(p_0=\lceil\sqrt{2(n+1)}\rceil=41\) to the full \(p=n=800\) improve the
original signed Max-Cut scores over **100 independent EVA initializations**?

This extends [Experiment028](experiment028-eva-fullrank-pilot.md), which
compared 5 EVA seeds and three rounding algorithms. Here we study **only**
`K-MeansNM_EVA`; no GW or K-Means2N runs are repeated.

## Fixed experiment design

- Graphs: G1 random/positive, G6 random/signed, G11 toroidal/signed,
  G14 planar-family/positive, G18 planar-family/signed. These family labels
  describe the G-set source generators. All have 800 vertices.
- EVA factor widths: **default p=41 versus full p=800**.
- 100 EVA initialization seeds **2026–2125 inclusive** for each graph and rank.
- **Do not re-solve** seeds 2026–2030: use Experiment028's **50 existing,
  checksum-verified saved factors and already scored K-MeansNM partitions**.
  Re-verify the originals' graph/factor/partition SHA-256, vector shapes, objective,
  and exact signed cut before importing them. Only the other **95 seeds per graph
  and rank** (950 factor solves) are new.
- Numba compiled EVA, float64, sequential Gauss–Seidel, `max_sweeps=1000`,
  angular tolerance `0.001` rad, checkpoints every 25 sweeps. Exact source
  `src/maxkcut/eva_numba.py` and its original checkpoint code remain unchanged.
  Full convergence is recorded, **not** assumed.
- K-MeansNM: **one antipodal spherical two-cluster initialization**, up to 100
  Lloyd iterations, retaining the highest original signed cut visited, with
  unchanged `round_spherical_kmeans`. Rounding RNG matches the previous
  experiment exactly:
  `numpy.random.SeedSequence([eva_seed, int(graph[1:]), 2, 2, 28])`.
- Zero Flip/KL/ILS refinements, no alternative rounders, and no convex SDP solve.
- Five independent graph-level processes on the single Windows self-hosted
  PCAG103A-13. BLAS/Numba threads set to one each. No old VM worktree changes.

Total design: **5 graphs × 100 EVA seeds × 2 ranks × 1 clustering method =
1000 signed hard cuts** from 1000 distinct EVA factors, comprising 50 reused
and 950 newly solved. Report the raw EVA continuous objectives separately from
rounded integer scores: neither is a certified global SDP optimum.

## Hypothesis and analysis

For each graph and seed define the paired contrast

\[
\Delta_{g,s}=C_{g,s}(p=800)-C_{g,s}(p=41),\quad
C_{g,s}(p)=\sum_{(i,j)\in E_g}w_{ij}\mathbf1\{\ell_i(p)\ne\ell_j(p)\}.
\]

Report per-graph means and standard deviations, paired mean differences,
100-seed best scores, wins/ties/losses, EVA means and timings. Include a
95% percentile bootstrap confidence interval for the paired seed mean,
using 5000 resamples and random seed 29029, **conditional on each selected
graph**. Also report the unweighted mean of five per-graph percent changes.
The seed-label pairing is reproducible but does not imply identical initial
embeddings across the different factor widths; it is a *matched seed identity*,
not a common-random-numbers coupling of equivalent geometries. With five graphs,
avoid claims about the broader G-set population or algorithmic optimality.

## New archival layout

`results/gset/experiment029-eva-kmeansnm-100seed/` contains:

- `all_runs.csv`: 1000 original signed cut scores, provenance and runtime.
- `paired_differences.csv`: 500 paired comparisons.
- `per_graph.csv`, `summary.json`, `report.md`: 100-seed analysis.
- `runs/Gxx/{default,full}/seed-<2031..2125>/factor.npz`: new full
  double-precision EVA vectors and checkpoint RNG states. Large objects use
  **new, scoped Git LFS rules**; no historical factor is moved or overwritten.
- `runs/.../partitions.npz`: original signed 2-cut label vector, packed and
  SHA-256 checked. It is never inferred from the displayed score.
- `runs/.../eva_sweeps.jsonl`: each EVA sweep objective/angular change/time.
- `runs/.../case.json`: seed, parameters, graph/partition/factor hashes,
  original objective and runtime metadata.

The first five seeds retain **references to original Experiment028 paths**,
rather than copying 50 binary objects into the new directory. All results must
be reproducible using both Experiment028 and Experiment029 together.

## Runner safety and checkpoint publication

- Use only GitHub Actions on **PCAG103A-13** after hosted tests and read-only
  input/hash validation succeed; use the Actions checkout, **never** the
  active `C:\Users\175199\maxkcut` Experiment025 checkout.
- The compute job cannot change the existing results branch. It uploads five
  separately retrievable graph archives, plus compact text analyses. If
  interrupted, preserved checkpoint files can resume only with the same
  exact graph/settings/seed/backend identity.
- A separate hosted publisher (no numerical solve) independently verifies
  original signed scores and SHA-256 for **all 950 new factors and partitions**,
  and commits the compact summary first, then five rank-factor archive batches
  with Git LFS. Large-files publication may fail because of quota; in that
  event, do not claim durable completion until a verified alternative is saved.
- GitHub Actions artifact retention alone is not durable storage.
- Any Windows text CRLF must be normalized to LF **only for text files**, never
  changing `.npz`/factor bytes.
- The new automated workflow fires **once on the specific workflow-file push**.
  Later documentation/result commits must not rerun computation. After completion
  retain only an explicit manual re-run trigger, if any.
- No merge without explicit authorization; PR will target the existing
  Experiment028 branch to preserve the unmerged source history.

## Commands, from a separate isolated checkout

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
python scripts/experiment029_kmeansnm_100seeds.py --validate-only
python scripts/experiment029_kmeansnm_100seeds.py --workers 5
python scripts/experiment029_kmeansnm_100seeds.py --check-only
```

Run the tests without numerical G-set computation:

```bash
PYTHONPATH=src python -m pytest tests/test_experiment029_kmeansnm_100seeds.py -q
```

**Do not repeat completed Experiment028 or Experiment029 EVA solves to
regenerate figures, reports, tables, or analysis.**