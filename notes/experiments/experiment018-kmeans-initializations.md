# Experiment018 — Initialization alternatives for two-center K-MeansNM

## Goal

The historically best observed G63 signed Max-Cut remains 26,917
(using KMeansNM-Flip+KL+ILS in Experiment013). Experiment017
showed no advantage for k>2 in its small first pilot:
k=2: 26,859; k=4: 26,843; k=6: 26,807;
k=8: 26,832; k=10: 26,804 after KL.

Experiment018 focuses on **two-centroid K-Means**, testing whether
better ways to pick the initial two directions improve the result.
It reuses the completed original G63 EVA Y (7000x119), unchanged,
and the original signed weighted graph. No new EVA/SDP solve.

## Nine initializers and experimental controls

| Name | Initialization | Extra initial graph-cut evaluations |
|---|---|---:|
| paired | Original Gaussian u, -u (K-MeansNM) | 0 |
| independent | Two independent Gaussian unit directions (K-Means2N) | 0 |
| sample-antipodal | Select a random actual EVA row y_i and its negative | 0 |
| sample-pair | Select two distinct random observed EVA rows | 0 |
| farthest-pair | Sample one EVA row, choose its farthest cosine-distance row | 0 |
| spherical-pp | Select one random EVA row; second with D² probability proportional to squared Euclidean distance on unit sphere | 0 |
| mean-perturb | Perturb EVA unit-mean in opposite orthogonal tangent directions; not opposite centers | 0 |
| pca-antipodal | Use leading centered EVA principal direction + seeded Gaussian jitter, then its opposite | 0 |
| gw-screen16 | Score 16 random candidate directions on ORIGINAL signed Max-Cut, choose highest; initialize antipodal | **16** |

All strategies then perform the SAME Lloyd assignment and normalized
centroid M-step. They never directly change the saved EVA matrix.
At each Lloyd iteration, retain the best real signed Max-Cut
partition, not necessarily the last geometric clustering.

Every raw candidate receives the same generalized fast KL budget.
The tests independently verify the exact original k=2 antipodal
seed schedule; the other strategies get the same restart/iteration
allocation. Scores are auditable using the ORIGINAL G63 signed
weights, not absolute edge values. The comparison controls for
restart count and KL passes but **GW-screen16 uses 16 extra
hard-cut evaluations, and PCA has an additional eigendecomposition**;
compare wall time and evaluation counts, not only best scores.
No claim that an alternative is superior until actual original
signed G63 cuts show an improvement. The full KMeansNM-Flip+KL+ILS
historical record is a stronger comparator than these initial
raw+KL runs.

## Runner: university Windows PCAG103A-13

Run automatically from the dedicated push workflow on this branch:

.github/workflows/experiment018-self-hosted.yml

It selects the already-installed `maxkcut` Conda Python, includes
the Conda Library/bin DLL path, checks the original archived
EVA file and graph, runs the k=2 legacy/alternatives regression
tests, then a small **9 strategies × 3 restarts** G63 pilot with
60 Lloyd iterations and 25 KL passes. The workflow always
publishes results as downloadable GitHub Actions artifacts and
commits successful results to its own experiment branch
(never main). The source EVA file remains immutable.

Manual PowerShell equivalent from repo root:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location).Path "src"
python -m pytest tests/test_eva_kmeans_initializers.py tests/test_eva_multigroup_kmeans.py -q
python -m maxkcut.eva_kmeans_init_cli `
    --graphs G63 `
    --methods paired independent sample-antipodal sample-pair farthest-pair spherical-pp mean-perturb pca-antipodal gw-screen16 `
    --restarts 3 `
    --chunk-size 1 `
    --workers 3 `
    --max-iter 60 `
    --top-merges 2 `
    --kl-passes 25 `
    --target 27047
```

For a larger run, use `--restarts 30 --chunk-size 1
--workers 4 --max-iter 100 --top-merges 4 --kl-passes 40`.
Changing budget values creates different settings fingerprints
rather than silently reusing results.

## Files and provenance

Results are committed under
`results/gset/experiment018-kmeans-initializations/`.

- `summary.csv`: all restart × method raw and KL scores.
- `leaderboard.json`: sorted best/mean final signed G63 cut.
- `groups/`: each retained two-way cluster labeling with exact
  group-merge mask and original per-iteration provenance.
- `cases/`: deterministic per-job settings IDs and full
  initialization metadata (including 16 extra evaluations).
- `trials/`: complete per-iteration diagnostic traces.
- `archives/`: independently verified raw/pre-KL/final and
  per-candidate bit-packed partitions on the original signed graph.
- `factor-snapshots/`: SHA256 exact source EVA factor.
- `errors/`: job failure diagnostics, retained for debugging.

The reported numerical target 27,047 is not a certified bound or
known optimum. A novel initializer cannot by itself guarantee that
KMeans escapes a geometrically stable partition; the objective is
to measure what differences in initial centroids actually induce.
