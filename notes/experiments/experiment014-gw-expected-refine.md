# Experiment014 — Reoptimize the EVA matrix for expected raw GW rounding

## Motivation

Latest user-observed G63 feasible Max-Cut: **26,917** from Experiment013.
Target: **27,047** (130-point remaining gap). This target is not a
certified maximum or upper bound.

The user's new idea is to **optimize the matrix itself once more**,
this time specifically for the quality of raw hyperplane rounding,
with NO K-means. The original EVA factor stays unchanged and serves as
a full-space warm start (G63: 7,000 vertices x 119 dimensions).

## Mathematics and distinct objectives

Original EVA vector relaxation maximizes

    0.5 sum_{i<j} w_ij [1 - y_i dot y_j].

This does NOT directly maximize GW rounding quality.

For the random Gaussian g, binary labels determined by the sign
of y_i dot g yield:

    Prob(different binary labels) = arccos(y_i dot y_j) / pi.

Thus for the original signed G-set weighted graph, optimize exactly

    E[GW Max-Cut] = sum_{i<j} w_ij arccos(y_i dot y_j) / pi.

This is an ANALYTICAL EXPECTED CUT, not the empirical best cut.
Negative graph weights are included with their original signs.
The unit-norm constraints on all 7,000 factor rows are preserved.

A numerical alternative epsilon>0 uses the precisely specified
independent Gaussian noisy rounding

    z_i = sqrt(1-epsilon)*(y_i dot g) + sqrt(epsilon)*eta_i

with eta_i independent standard Gaussian. The corresponding pairwise
correlation is (1-epsilon)(y_i dot y_j), so the expected noisy GW cut
has the same arccos formula with the correlation multiplied by
(1-epsilon). With epsilon=0 this is EXACTLY ordinary raw GW.

## Optimizer

- Numba gradient over sparse undirected signed edges, no dense graph.
- Sphere-tangent projected ascent with backtracking and norm checks.
- Only ACCEPT steps that INCREASE exact expected signed cut.
- Optional seeded tangent perturbations to escape a local basin.
- Always preserve the best starting matrix: best expected cut after
  must be >= original EVA matrix's expected cut.
- Store the newly optimized full matrix, complete per-sweep objective
  traces, inputs and hashes separately from original EVA factors.
- This is nonconvex and does NOT guarantee a better realized cut.

## Paired experiments (NO K-means)

1. OldRawGW: raw Gaussian hyperplanes on the original EVA matrix.
2. NewRawGW: IDENTICAL hyperplanes on new expected-GW-trained matrix.
3. OldRawGWKL: KL on selected original raw cuts.
4. NewRawGWKL: KL on the matched new raw cuts.

For epsilon>0, both matrices also share identical per-vertex Gaussian
noise in each paired trial. Select the same KL draw indices based on
the maximum of the two RAW scores, not one side. Independent KL
refinement never decreases its own raw input score.

The original signed graph Max-Cut score of EVERY stored raw and KL
partition is independently verified. Report BOTH continuous expected
GW gain and actual feasible raw/KL score gain.

## Windows PowerShell runbook

First, on the existing Experiment013 branch, preserve results:

    git add .
    git commit -m "experiment013: preserve G63 grid best 26917"
    git push

Wait for a successful push, then change branches:

    git fetch origin
    git switch --track origin/experiment014-gw-expected-refine

If already tracked, use git switch experiment014-gw-expected-refine.

The original EVA factor is expected at
results/gset/experiment003-eva-numba/factors/G63.npz.
If it is not visible after switching, restore the committed original
factor file from its proper results branch; do not re-run EVA.

Run correctness tests FIRST:

    $env:PYTHONPATH = Join-Path (Get-Location).Path "src"
    python -m pytest tests/test_eva_gw_expected.py -q

## Small pilot: two matrix reoptimizations

    python -m maxkcut.eva_gw_expected_cli --graphs G63 --epsilons 0 --jitters 0 0.05 --steps 0.05 --optimize-seeds 2026 --max-sweeps 20 --patience 8 --rounds 150 --kl-top 10 --kl-passes 20 --workers 2 --target 27047

## Large 16-configuration matrix optimization grid

    python -m maxkcut.eva_gw_expected_cli --graphs G63 --epsilons 0 0.02 --jitters 0 0.05 --steps 0.05 0.15 --optimize-seeds 2026 2027 --max-sweeps 60 --patience 12 --rounds 1000 --kl-top 30 --kl-passes 35 --workers 4 --target 27047 --push --push-branch experiment014-gw-expected-refine

16 independent expected-GW matrix optimizations. Each has 1,000
matched raw Gaussian hyperplanes on both matrices and 30 paired KL
draws: 32,000 original/new raw binary partitions, 960 KL-refined
binary partitions, with the 16 full optimized matrices retained.

More extensive budgets: --max-sweeps 150 --rounds 3000 --kl-top 80.
Changing settings creates a NEW case and partition/factor archive,
without overwriting any prior results. Identical reruns verify and
skip complete outputs rather than recomputing.

## Repo-first complete outputs

- factor-snapshots/: exact unchanged original EVA factor and SHA.
- optimized-factors/: separate optimized full n-by-p matrix per
  config, norm-checked and SHA-verified on replay.
- cases/: JSON config, expected objective history, full raw and KL
  scores, source/optimized factor hashes.
- archives/: original weighted graph snapshot and ALL raw/KL
  bit-packed signed Max-Cut partitions with independent verification.
- trials/: individual raw paired draw scores and KL paired scores.
- summary.csv: hyperparameter grid, expected gain, raw and KL best
  scores, target gap and artifact paths.
- errors/: failures and traceback diagnostics.

Use --push --push-branch experiment014-gw-expected-refine for ONE
repository push at end. Or manually commit and push completed output.

No numerical G63 result beyond 26,917 is claimed before the user's
own Windows VM tests and objective verification. A feasible 27,047
cut would not establish the global Max-Cut optimum.
