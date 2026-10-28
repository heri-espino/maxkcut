# Experiment012 — Improve K-MeansNM rounding before Kernighan–Lin

## Motivation

Observed feasible G63 signed weighted Max-Cut values:
- EVA K-MeansNM + 1,000 local-flip roundings: 26,682.
- Experiment010's generalized KL on 200 archived starts: 26,886.
- Numerical target: 27,047 (161 above 26,886), NOT a certified optimum.

Experiment011 concentrates on diversifying local KL solutions. This
experiment changes the earlier rounding stage instead. It keeps the
completed, verified EVA factor fixed (for G63, R^119); no new EVA
optimization, SDP, PCA or MST construction takes place.

## Mathematical method

Original antipodal 2-center spherical K-MeansNM has zero-offset
hyperplane classification, with projection

    z_i = y_i dot (c_1 - c_0).

Its objective is cluster geometry, not the original signed Max-Cut
score sum_{(i,j) in E} w_ij * 1[labels_i != labels_j].

The NEW procedure:
1. Reproduce the original deterministic antipodal K-MeansNM initial
   centers and Lloyd update trajectory, tracking original weighted
   graph cut at every stage.
2. For each projected direction, sort all n values of z_i and sweep
   the original signed graph incremental flip gains. This finds
   EXACTLY the best threshold cut among ALL n+1 possible scalar
   threshold partitions along that fixed direction. Equal projection
   values are moved as an inseparable group. Cost O(n log n+m)
   once projected, with no dense adjacency matrix.
3. Explore additional directions by perturbing the incumbent, using
   centroid differences and original-graph signed flip-gain weighted
   centroid proxies. Every candidate's actual Max-Cut is scored;
   its best threshold is found exactly. Direction search is a
   heuristic, not an exact global optimization.
4. Run the existing generalized Numba indexed-heap KL optimizer on
   the best cut-aware K-Means partition.
5. For a paired baseline, separately run KL on the EXACT same archived
   K-MeansNM-Flip partition, using experiment010's KL seed schedule.

Four complete partitions and signed scores are independently saved
for every identical starting restart:
- KMeansNMFlipInput
- CutAwarePreKL
- KMeansNMFlipKL (paired direct-KL control)
- CutAwareKMeansKL (proposed enhanced K-Means then KL)

The final proposed cut is guaranteed to be >= its own CutAwarePreKL
cut; the direct-KL control is guaranteed >= its own archived input.
A threshold-optimized geometric cut need not beat an already
individual-vertex-flip-optimized baseline, since they use different
search spaces. Compare paired outcomes before claiming improvement.

## PowerShell: preserve and switch

Before switching from the branch holding experiment008/010 results:

    git add .
    git commit -m "save: completed G63 experiment008-011 outputs"
    git push

Wait for the successful push, then:

    git fetch origin
    git switch --track origin/experiment012-cut-aware-kmeans-kl

If already created locally, use git switch experiment012-cut-aware-kmeans-kl.
If the input archives disappear after the switch, merge or restore
their committed results from the exact source branch. DO NOT rerun EVA.

## Tests

    $env:PYTHONPATH = Join-Path (Get-Location).Path "src"
    python -m pytest tests/test_eva_cutaware_kmeans.py tests/test_eva_elite_search.py -q

## Pilot: 40 selected starts, 48 new direction proposals each

    python -m maxkcut.eva_cutaware_kmeans_cli --graphs G63 --factor-dir results/gset/experiment003-eva-numba --rounding-dir results/gset/experiment008-eva-rounding-flips --output-dir results/gset/experiment012-cut-aware-kmeans-kl --top-starts 40 --chunk-size 10 --workers 4 --kmeans-iters 100 --direction-trials 48 --kl-passes 30 --target 27047

## More extensive: 200 starts, 256 direction proposals each

    python -m maxkcut.eva_cutaware_kmeans_cli --graphs G63 --factor-dir results/gset/experiment003-eva-numba --rounding-dir results/gset/experiment008-eva-rounding-flips --output-dir results/gset/experiment012-cut-aware-kmeans-kl --top-starts 200 --chunk-size 10 --workers 4 --kmeans-iters 100 --direction-trials 256 --kl-passes 30 --target 27047

## Intensive: ALL 1000 starts, 512 proposals, 50 KL passes

    python -m maxkcut.eva_cutaware_kmeans_cli --graphs G63 --factor-dir results/gset/experiment003-eva-numba --rounding-dir results/gset/experiment008-eva-rounding-flips --output-dir results/gset/experiment012-cut-aware-kmeans-kl --top-starts 0 --chunk-size 10 --workers 4 --kmeans-iters 100 --direction-trials 512 --kl-passes 50 --target 27047 --push --push-branch experiment012-cut-aware-kmeans-kl

This high-budget experiment can be computationally substantial
despite fast individual cut evaluations. It does not rerun the EVA
optimizer or the 1,000 original roundings.

## Provenance and append-only reproducibility

Inputs are the verified experiment008 bit-packed partitions, completed
original-space EVA factor and original G-set weighted graph, all with
SHA256 checks. Duplicated restart indices and mismatched factors fail.
Exact same settings and chunks verify and skip saved results.
Changing settings creates NEW case identities, rather than overwriting
old artifacts.

Outputs under results/gset/experiment012-cut-aware-kmeans-kl:
- cases/: per-chunk settings, original scores, all four stage scores.
- factor-snapshots/: SHA-verified immutable factor.
- archives/: original weighted graph and independently rescored
  bit-packed binary partitions for every restart and stage.
- trials/: per-restart JSONL direction and KL diagnostics.
- summary.csv: per restart and method before/after and gap to 27,047.
- summaries/: best/mean/median/std per stage, by experiment settings.
- errors/: graph and per-chunk traces.

Any result >= 27,047 is a feasible Max-Cut found by a heuristic,
not a proof of Max-Cut optimality.

## Future direction

If threshold optimization helps, run experiment011's iterated KL
and complement-aligned elite recombination using the new
CutAwareKMeansKL partitions as additional elite starts. Always
keep signed objective and matched compute budgets in comparisons.
