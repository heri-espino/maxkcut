# Experiment015 — Optimize only the RAW GW rounding normal (exact sweeps and stochastic gradients)

## Scientific question

From the already completed G63 EVA factor Y in R^(7000x119), can we
directly optimize the *rounding hyperplane* rather than using K-means
or optimizing Y itself? Best user-observed signed weighted G63
Max-Cut is **26,917** (Experiment013); the target **27,047** is
130 points higher. 27,047 is a target, NOT a certificate or known
upper bound.

The optimized variable is ONLY r in R^119, ||r||_2=1:

    C(r) = sum_{i<j} w_ij * 1[(y_i dot r > 0) != (y_j dot r > 0)].

Keep Y FIXED. No new EVA, SDP, Gram factorization, PCA, MST,
or K-means. The search is restricted to hyperplane-separable
partitions. The optional KL postprocessor may leave this restricted
family, so a better post-KL cut can exceed the best geometric cut.

## 1. Exact great-semicircle line search (no surrogate)

For any current unit normal r and perpendicular unit tangent v:

    r(theta) = cos(theta) r + sin(theta) v, -pi/2<theta<pi/2.

Vertex i flips only once along the arc when

    theta_i = atan( -(y_i dot r)/(y_i dot v) ),

provided y_i dot v != 0. Sort the n crossing angles, flip all
equal-angle events together (a threshold at a tie cannot separate
tied vertices), and maintain exact signed graph flip-gain increments
through sparse original graph CSR adjacency. The best interval is
optimal over this ONE great semicircle, in O(n log n + m) plus
the two EVA projections; it is not globally optimal over the whole
unit sphere. Repeat the exact sweep with random, signed-gain-guided
and small-flip-gain-boundary tangents. Preserve the BEST verified
hard original-graph cut after every step.

## 2. Gaussian-smoothed stochastic gradient ASCENT (ES-SGA)

A true hard rounding cut C(r) is piecewise constant, so ordinary
differentiation gives zero gradient away from discontinuities.
Instead define

    F_h(r) = E_eps[C(r+h*eps)], eps~N(0,I_p).

Although C itself is discontinuous, F_h is differentiable, and
an ANTITHETIC unbiased Euclidean gradient estimate is

    grad F_h(r) = E[(C(r+h eps)-C(r-h eps))*eps] / (2h).

Use B Gaussian perturbations per step, B user-controlled, and
h=angular_noise/sqrt(p). Sample the ACTUAL original signed G-set
weighted Max-Cut values for each pair, use that gradient estimate
with Adam, project it to the sphere tangent space, renormalize r,
and anneal noise strength. All probed directions and the Adam
candidate can become the BEST saved hard cut. When the search is
stagnant, jump to a perturbed incumbent neighborhood. Adam's
current iterate can worsen, but the recorded feasible incumbent
never worsens. We do NOT claim the stochastic trajectory monotonically
increases F_h or attains the global hard optimum.

This procedure is gradient-based stochastic optimization of the
SMOOTHED EXPECTATION OF THE ORIGINAL HARD MAX-CUT, not K-means,
a new EVA matrix solve, or a deterministic gradient of C.

## 3. Independent signed-edge minibatch soft-SGA baseline

Compare against a more conventional differentiable stochastic
relaxation with p_i=sigmoid(y_i dot r / tau) and

    SoftCut_tau(r) =
        sum_{i<j} w_ij [p_i+p_j-2*p_i*p_j].

This is EXACTLY the expected cut under INDEPENDENT Bernoulli node
assignments, not the original shared-hyperplane GW distribution.
Uniformly sample edges with replacement and use the unbiased
minibatch gradient of this soft signed-weight objective, with Adam
and decreasing temperature tau. Score and archive the best
DETERMINISTIC hard hyperplane cut C(r), not its soft objective.

This is informative as an alternative SGD loss, but it may perform
worse because its independent Bernoulli surrogate differs from the
shared-hyperplane rounding objective.

## Methods compared, same initial GW normals

- linesweep: exact line sweeps from the initial FJ/GW normal.
- es-sga: antithetic Gaussian stochastic ascent.
- es-sga-sweep: SAME seeded ES-SGA run, then exact line sweeps.
- soft-sga: signed-edge minibatch logistic surrogate ascent.
- soft-sga-sweep: SAME seeded soft-SGA run, then exact line sweeps.

Each method starts from exactly the same raw FJ/GW direction for
each original restart number, generated with the original source
seed schedule SeedSequence([rounding_seed, graph_number, 2, restart, 0]).
Each method's output archive includes FOUR original signed Max-Cut
partitions and scores:

    GWRaw: same initial raw hyperplane;
    OptimizedRaw: optimized normal on unchanged Y;
    GWRawKL: original raw hyperplane then generalized KL;
    OptimizedRawKL: optimized raw hyperplane then generalized KL.

The KL routines have the SAME per-restart/replicate RNG stream for
old/new labels and independently preserve each starting cut.
The optimized direction vectors themselves are saved in complete
119-dimensional NPZ arrays with SHA256 and verified against the
reconstructed archived hard labels.

The procedure is stochastic, nonconvex and can stagnate. No
direct-rounding or stochastic run is guaranteed to reach 27,047.

## Windows PowerShell runbook

First preserve Experiment013's G63 result 26,917 and any subsequent
outputs on the CURRENT branch; do not change branches while an
experiment is writing files:

    git add .
    git commit -m "experiment013: preserve G63 best 26917"
    git push

Then switch to the implementation branch:

    git fetch origin
    git switch --track origin/experiment015-direct-hyperplane-cut

If this branch already exists locally:

    git switch experiment015-direct-hyperplane-cut

The original factor must be present in
results/gset/experiment003-eva-numba/factors/G63.npz,
alongside its completed EVA case JSON. If the archived source
files are on a different pushed branch, merge or restore those
verified files; DO NOT rerun the expensive original EVA solver.

### First run correctness and Windows multiprocessing tests

    $env:PYTHONPATH = Join-Path (Get-Location).Path "src"
    python -m pytest tests/test_eva_direct_rounding.py -q

### Pilot: test exact search and both stochastic gradient approaches

    python -m maxkcut.eva_direct_rounding_cli --graphs G63 --methods linesweep es-sga es-sga-sweep soft-sga soft-sga-sweep --start-restart 0 --restarts 10 --chunk-size 2 --workers 4 --line-directions 40 --sga-steps 100 --sga-population 8 --sigma-start 0.3 --sigma-end 0.03 --edge-batch 1024 --temperature-start 0.5 --temperature-end 0.05 --learning-rate 0.08 --restart-patience 40 --kl-passes 25 --kl-repeats 2 --target 27047

This executes 5 methods x 10 exactly paired initial normals, with
the stochastic optimizers allocating up to 100 updates each.
It saves both original and optimized raw partitions plus KL
postprocess results; no matrix reoptimization or K-means.

### More extensive: 100 starts, 800 stochastic steps, exact sweeps

    python -m maxkcut.eva_direct_rounding_cli --graphs G63 --methods linesweep es-sga es-sga-sweep soft-sga soft-sga-sweep --start-restart 0 --restarts 100 --chunk-size 2 --workers 4 --line-directions 100 --sga-steps 800 --sga-population 24 --sigma-start 0.35 --sigma-end 0.01 --edge-batch 4096 --temperature-start 0.5 --temperature-end 0.02 --learning-rate 0.06 --restart-patience 100 --kl-passes 50 --kl-repeats 3 --target 27047 --push --push-branch experiment015-direct-hyperplane-cut

Because budgets differ between the pilot and extended run, their
exact configuration fingerprints differ and both remain archived.
To reuse completed chunk results, rerun the same settings and
increase only --restarts, keeping --start-restart and --chunk-size
unchanged. Each verified chunk then skips without recomputation.

## Outputs and scientific interpretation

results/gset/experiment015-direct-rounding contains:

- factor-snapshots/: original EVA factor, copied with SHA check.
- normals/: RAW original and optimized hyperplane normals, plus
  original restart indices, saved for ALL methods and configurations.
- cases/: settings fingerprints, per-restart score, trajectories,
  selected normal and original graph/factor SHA256.
- archives/: original sparse weighted graph snapshot and complete
  bit-packed binary GWRaw / OptimizedRaw / GWRawKL /
  OptimizedRawKL partitions, independently rescored.
- trials/: complete per-restart JSONL stochastic/sweep traces,
  signed actual cut gains and multiple KL attempt scores.
- summary.csv: paired raw and post-KL cut scores and gaps to 27,047.
- errors/: failure diagnostics with traceback and method/chunk IDs.

Optimization happens in independent Windows-safe processes.
Only finished chunks are resumable; partially interrupted chunks
need recomputation. Git push is opt-in once at the end or manual:

    git add .
    git commit -m "experiment015: archive direct stochastic GW rounding"
    git push

Report original signed graph Max-Cut after KL separately from raw
hyperplane Max-Cut. A KL improved partition may not be realizable
by any single hyperplane in the fixed EVA embedding. 27,047 remains
a numerical target, not a certified optimum. Existing user-reported
best remains 26,917 unless new verified cuts exceed it.
