# Experiment016 — Soft EM + Bernoulli CEM / Gibbs / KL

## Hypothesis

G63 currently has a user-verified feasible cut **26,917**. The target
**27,047** is 130 points higher; it is not a certified optimum.

After Experiment015 directly optimized a *single hyperplane* (raw
scores improved, but its reported post-KL results did not exceed
26,917), we now explore a genuine **distribution of partitions**.
No new EVA matrix solve, no SDP/Gram recomputation, and no hard
K-Means center updates. We use the original full 7000x119 EVA factor,
the original signed weighted graph, and optionally the best actual
Experiment013 partition as an independently verified warm start.

## Mathematical formulation: proper soft alternating optimization

Let Y_i be unit vectors, and x_i~Bernoulli(p_i). For two unit
centroids c0 and c1 and entropy temperature T, maximize

    F(p,c0,c1;T)
      = sum_{i<j} w_ij (p_i+p_j-2*p_i*p_j)
        + kappa sum_i [(1-p_i)Y_i·c0+p_i Y_i·c1]
        + T sum_i [-p_i log p_i-(1-p_i)log(1-p_i)].

This is **expected original signed Max-Cut of independent
Bernoulli assignments** plus an EVA geometric term and entropy.
The signed original weights are never discarded or made absolute.
At fixed centroids and all other probabilities, a coordinate's
conditional maximum is exactly

    p_i = sigmoid(
       [sum_j w_ij(1-2p_j)
        + kappa Y_i·(c1-c0)] / T
    ).

The implementation takes damped, SYNCHRONOUS steps toward these
coordinate optima and only accepts a step if the common F objective
does not decrease. It then updates both spherical centroids by exact
unit normalization of the probability-weighted vector sums.
Because T changes between annealing stages, ascent is audited
*within each stage*, not claimed across different temperatures.
This is **EM-like / variational mean field alternating optimization**,
not statistical maximum-likelihood EM for a fitted generative
mixture model. The Bernoulli probabilities remain soft until
evaluation or downstream stochastic sampling.

### Initializations

- mean: use the normalized mean of all EVA vectors, perturb it in a
  random tangent direction to create two different centroids; if
  the empirical mean is zero, use a reproducible random base.
- elite: use the SAME random mean-centroid perturbation but initialize
  probabilities near a verified known binary Max-Cut partition and
  recalibrate spherical centroids. This is important in 7000D
  Bernoulli state space: an unbiased p≈0.5 has effectively no chance
  of sampling a high-quality cut with a modest population.

Every episode has a separate deterministic seed. For elite mode,
the loader ranks actual Experiment013 *Grid-Final_EVA* outputs,
verifies archive against the original G63 file SHA256, verifies
factor SHA256, independently re-scores every accepted elite, removes
complement-duplicate cuts and never trusts a reported 26,917
without checking its bit-packed labels and graph cut.

## Four complete pipelines on matched starts

1. **em-kl**: annealed soft EM probabilities; threshold p>=0.5,
   retain best of that and any verified input elite, then compiled
   generalized KL.
2. **em-cem**: soft EM + Cross-Entropy Method (CEM) on factorized
   Bernoulli probabilities; no KL.
3. **em-gibbs**: soft EM + single-site Gibbs / simulated annealing
   on ORIGINAL signed Max-Cut graph; no KL.
4. **em-cem-kl**: SAME seeded CEM run as em-cem, plus compiled KL.

CEM samples B independent Bernoulli binary partitions per generation,
evaluates their REAL original signed graph cut, selects the top
fraction as elites, complement-aligns each selected partition with
the global best current partition, and updates the probability
vector by exponential smoothing of elite vertex frequencies. A
nonzero minimum probability and entropy injection preserve some
exploration; strong initial anchor weighting protects the known
high-dimensional solution from random destruction. Retain
the best *feasible hard* cut, not the sampling distribution mean.

Gibbs annealing repeatedly visits vertices in random order. For
spins s_i=1-2x_i, the exact gain of flipping vertex i is

    Δ_i = s_i sum_j w_ij s_j.

At temperature T, flip with probability sigmoid(Δ_i/T).
This is a heat-bath single-variable Gibbs transition for the
distribution P(x)∝exp(C(x)/T). The Markov chain current cut
can decrease; the best original graph cut is ALWAYS retained.
The implementation audits the incremental gain against a full
cut computation after each sweep.

KL is treated as an independent final control. CEM and Gibbs are
NOT guaranteed to beat KL: comparisons measure how many actual
points of improvement remain after KL, and relative CPU budgets.

## PowerShell on the existing Windows VM

### 1. Preserve Experiment015 (and Experiment013) results

On the existing local working branch, commit/push completed work:

    git add .
    git commit -m "experiment015: save direct-rounding G63 results"
    git push

Then switch to the experiment016 implementation:

    git fetch origin
    git switch --track origin/experiment016-soft-em-bernoulli

If already tracked, use:

    git switch experiment016-soft-em-bernoulli

**Important**: Confirm these completed original inputs are present:

- results/gset/experiment003-eva-numba/factors/G63.npz
- results/gset/experiment003-eva-numba/cases/G63.json
- results/gset/experiment013-eva-grid/cases/
- results/gset/experiment013-eva-grid/archives/

If Experiment013 results are on a different branch, bring the
previously pushed experiment013 result files to this branch. The
software will NEVER rerun original EVA to substitute for them.

### 2. Run regression/integration tests before G63

    $env:PYTHONPATH = Join-Path (Get-Location).Path "src"
    python -m pytest tests/test_eva_soft_em_bernoulli.py -q

Tests cover the common free-energy monotonicity within fixed T;
zero-mean centroids, deterministic mean/elite restarts; Bernoulli
complement alignment, CEM best-incumbent retention, signed Gibbs
incremental cut accounting and deterministic replay; independently
verified synthetic elite archive + Windows spawn ProcessPoolExecutor
+ source factor immutability + four saved partitions and probability
reconstruction + idempotent restart.

### 3. Small G63 pilot: 4 methods × 2 modes × 4 episodes

    python -m maxkcut.eva_soft_em_bernoulli_cli --graphs G63 --initializations mean elite --episodes 4 --chunk-size 2 --workers 4 --em-stages 4 --em-iters 8 --cem-generations 10 --population 48 --gibbs-sweeps 40 --kl-passes 25 --target 27047

This yields 32 matched pipeline episodes, first verifying the best
available distinct Experiment013 elite sources. Summary CSV and
complete full-partition/probability archives are written after
each finished chunk.

### 4. Expanded run: 4 methods × 2 modes × 16 episodes

    python -m maxkcut.eva_soft_em_bernoulli_cli --graphs G63 --initializations mean elite --episodes 16 --chunk-size 2 --workers 4 --em-stages 8 --em-iters 20 --cem-generations 40 --population 128 --elite-fraction 0.15 --update-rate 0.6 --anchor-weight 0.96 --injection 0.035 --gibbs-sweeps 200 --gibbs-temp-start 12 --gibbs-temp-end 0.4 --kl-passes 50 --target 27047

Because this increases the EM/CEM/Gibbs/KL **budget**, it uses new
configuration fingerprints. Completed pilot chunks will not be
mistakenly reused with different parameter settings. To resume the
SAME run, repeat the exact command. The output case verifies its
archive, original signed graph hash, source factor hash and actual
EM probabilities before skipping. To extend episode count without
changing other settings, keep chunk size and seeds fixed.

A separate simple test with only mean starts (no Experiment013 elite
folder) is available by using:

    python -m maxkcut.eva_soft_em_bernoulli_cli --graphs G63 --initializations mean --episodes 4 --workers 4 --em-stages 4 --em-iters 8 --cem-generations 10 --population 48 --gibbs-sweeps 40

## Saved artifacts (source of truth is the GitHub repository)

Directory: results/gset/experiment016-soft-em-bernoulli/

- factor-snapshots/: SHA256 exact copy of original EVA factor.
- cases/: full experiment settings and source archive hashes, signed
  graph SHA, per-episode and per-method trial results.
- probabilities/: exact float64 per-vertex EM Bernoulli probabilities
  and CEM-updated probabilities; SHA checked on resume.
- archives/: four independently checked bit-packed partition sets
  for each episode: Source, EMHard, Candidate, Final.
- trials/: per-episode JSONL EM temperature/entropy/free energy,
  CEM population/elite cut, Gibbs annealing and KL traces.
- summary.csv: every available result, raw, sampled and KL
  signed graph Max-Cut, gain vs verified source and gap to 27047.
- errors/: failure diagnostics; completed chunks remain reusable.

Once a run succeeds, push all experiment results:

    git add .
    git commit -m "experiment016: save G63 EM Bernoulli CEM Gibbs results"
    git push

Only an independently verified G63 cut above 26,917 is a new
record. A higher **soft expected** cut score does not prove any
improvement in the hard feasible maximum. This is not a global
Max-Cut algorithm/certificate; 27,047 remains a numerical target.
