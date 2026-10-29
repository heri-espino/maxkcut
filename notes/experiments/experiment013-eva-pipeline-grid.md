# Experiment013 — Grid search over complete EVA rounding and refinement pipelines

## Current verified user observations on G63

- EVA K-MeansNM + positive-gain vertex flips, 1,000 restarts: 26,682.
- Experiment010 generalized Kernighan–Lin after those flips: 26,886.
- Experiment012 cut-aware hyperplane K-Means then KL: 26,877.
- Numerical target: 27,047. The gap is **161** from the current
  user-observed best. It is NOT a certified optimum or upper bound.

Experiment012 demonstrated that a better geometric rounding method
may still produce a worse final KL solution: score attribution matters.
Experiment013 evaluates entire reproducible pipelines and retains ALL
of their intermediate and final partitions rather than proceeding
one novel method at a time.

## Grid design

The checked-in source-of-truth file is

    configs/experiment013-grid.json

It defines a list of method families; within each family, parameters
under "axes" are expanded in their full Cartesian product (defaults
and shared parameters may be supplied under "base"). The initial
checked-in grid contains **43 unique configurations** across:

- Saved archived K-MeansNM-Flip source + KL (including KL seed repeats).
- Reproduced raw antipodal spherical K-MeansNM + 0/20/100 flips and KL.
- Ordinary 2-center spherical K-Means2N + flips and KL.
- GW/FJ random hyperplanes + flips and KL.
- GW/FJ hyperplanes with globally optimal signed graph threshold
  along that one direction, plus flips and KL.
- Cut-aware spherical K-MeansNM, optimizing every fixed direction's
  scalar signed Max-Cut threshold exactly, with 0/48/256 further
  direction proposals, flips and KL.
- Saved K-MeansNM-Flip + KL + ILS basin escape.
- Cut-aware spherical K-MeansNM + flips + KL + ILS.

Unlike unbounded Cartesian expansion, the family structure avoids
unhelpful budget dimensions for methods that cannot use them.
Users can edit JSON to add/drop pipelines, increase budget levels,
or change the post-optimization schedule. Invalid combinations
and more than 500 configs in a single grid are rejected.

Stages for each paired source restart are:

    full original EVA embedding Y (G63: R^119)
        -> Round (6 families)
        -> optional positive-gain flips (0/20/100)
        -> optional generalized KL (0/15/30/40/60 passes)
             with optionally 1 or 3 independent seeded repeats
        -> optional iterated KL (0/20/40/80 ILS cycles)
        -> signed weighted G-set Max-Cut on ORIGINAL graph

For each configuration and starting restart, the full binary
partition at EVERY stage (Round / Flip / KL / Final) and its
original signed graph Max-Cut score is independently archived,
not merely the configuration's best. For all refiners,
the returned stage's score is nondecreasing relative to its own
earlier stage. A new rounded partition may score below another
pipeline's previous optimized cut: that's a normal experiment.

## Reproducibility / fairness / interpretation

This is NOT a new SDP or EVA solve. The source factor was optimized
once in experiment003. The following are read/verified:

- Graph original file SHA256.
- Exact saved EVA factor and its original optimizer provenance,
  factor rank, and immutable factor snapshot.
- Experiment008 saved bit-packed K-MeansNM-Flip partitions and scores,
  every source segment and partition NPZ SHA256. Duplicate input
  restart indices and mismatched SHA versions are hard errors.
- All 43 pipelines are evaluated on exactly the SAME selected restart
  indices (top 40 initially) from those 1,000 original cut scores.
  Raw K-Means, GW and KMeans2N methods regenerate their exact
  old per-restart rounding seed schedule, without EVA solves.
- KL seed streams and ILS streams are independently seeded and do
  not change with worker count, chunk size or job scheduling.
- SHA-fingerprinted, append-only per-configuration/per-chunk archives
  independently re-score EVERY saved stage on the original graph.
- Distinct budgets, target thresholds and seeds create new cache keys,
  preserving earlier results. Partial failed chunks leave completed
  chunks available for verification and resume.

The initial top-40 comparison is **score-selected** and therefore
not an unbiased estimate of out-of-sample method quality. To
investigate robustness, promote the strongest pipelines to top 200,
then all 1,000 source starts. We care primarily about finding a
higher feasible G63 cut, but should report mean and median across
paired starts, gains compared to the same baseline, and CPU seconds
to assess whether the method itself is an improvement.

The ranking sorts by best feasible Max-Cut, then mean, then compute
time (among tied scores). This is an EXPLICIT optimization goal;
the best of 43 methods is not a statistical guarantee.

## Windows PowerShell: save your completed Experiment012 first

Before switching, preserve the existing result artifacts to GitHub:

    git add .
    git commit -m "experiment012: save G63 objective-aware KMeans and KL results"
    git push

Then switch to this code branch:

    git fetch origin
    git switch --track origin/experiment013-eva-pipeline-grid

If already locally tracked:

    git switch experiment013-eva-pipeline-grid

If the factor and experiment008 archive inputs were committed to
another results branch, merge that pushed results branch into
experiment013 before running. Do NOT rerun EVA.

Test the grid implementation and its staged-parallel source archive
checks:

    $env:PYTHONPATH = Join-Path (Get-Location).Path "src"
    python -m pytest tests/test_eva_pipeline_grid.py tests/test_eva_cutaware_kmeans.py -q

## Stage 1: run ALL 43 configurations on the top 40 starts

    python -m maxkcut.eva_pipeline_grid_cli --graphs G63 --top-starts 40 --workers 4 --chunk-size 10 --grid configs/experiment013-grid.json --factor-dir results/gset/experiment003-eva-numba --rounding-dir results/gset/experiment008-eva-rounding-flips --output-dir results/gset/experiment013-eva-grid --target 27047

The command prints individual chunk progress, running global best
score and gap to 27,047, then writes a unique leaderboard JSON/CSV
under results/gset/experiment013-eva-grid/leaderboards/.
Every complete configuration has 40 paired original source restarts.

## Stage 2: promote best 8 configurations to top 200 starts

Use the exact JSON leaderboard from Stage 1 (the command prints the
path). For example, in PowerShell:

    $leaderboard = (Get-ChildItem results/gset/experiment013-eva-grid/leaderboards/G63-*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
    python -m maxkcut.eva_pipeline_grid_cli --graphs G63 --top-starts 200 --workers 4 --chunk-size 10 --grid configs/experiment013-grid.json --factor-dir results/gset/experiment003-eva-numba --rounding-dir results/gset/experiment008-eva-rounding-flips --output-dir results/gset/experiment013-eva-grid --target 27047 --promote-from "$leaderboard" --promote-top 8

Source IDs are prefixes of the same sorted original-restart ranking,
so Stage 2 REUSES and verifies the first four 10-restart chunks for
the promoted configurations. It only computes the new starts.

## Stage 3: promote best 3 to ALL 1,000 saved starts

Use the Stage 2 leaderboard JSON (not Stage 1). If necessary,
set the path manually rather than blindly selecting the newest file:

    $leaderboard = (Get-ChildItem results/gset/experiment013-eva-grid/leaderboards/G63-*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
    python -m maxkcut.eva_pipeline_grid_cli --graphs G63 --top-starts 0 --workers 4 --chunk-size 10 --grid configs/experiment013-grid.json --factor-dir results/gset/experiment003-eva-numba --rounding-dir results/gset/experiment008-eva-rounding-flips --output-dir results/gset/experiment013-eva-grid --target 27047 --promote-from "$leaderboard" --promote-top 3 --push --push-branch experiment013-eva-pipeline-grid

A value of --top-starts 0 means all available source restarts
(1,000 in the completed G63 experiment008). Re-running the identical
command verifies/safely skips saved chunks. Larger search budgets
are configurable by editing the checked-in JSON grid, which creates
new, reproducible configuration IDs.

## Output files

- configs/experiment013-grid.json: reproducible configurable methods.
- results/gset/experiment013-eva-grid/cases/: fingerprinted per-chunk
  status, configuration, restart IDs, stage scores and signed graph hash.
- archives/: per-chunk saved graph snapshot and ALL 4 sets of packed
  binary partitions, with independent re-scoring.
- factor-snapshots/: exact original EVA factor checksum copy.
- trials/: JSONL with timing, method, restart, direction and local
  optimization diagnostics.
- summary.csv: every archived trial/stage in easy tabular form.
- leaderboards/G63-<fingerprint>.json and CSV: per-selected-input
  best/mean/median/std, signed cut gains, compute cost and target gap.
- errors/: failure tracebacks for resumable diagnosis.

Git is the long-lived source of truth; a VM can be discarded after
successfully committing and pushing all results.

## Further directions

If the broad grid improves little over KL=26,886, the likely
bottleneck is not the choice of a geometric hyperplane. Prioritize
iterated KL on a richer ELITE pool and exact small-subset ILP
neighborhoods, maintaining signed graph scoring and independent
verification. A score of 27,047 on G63 would be a feasible cut,
not a proof of a global maximum without a certified upper bound.

## Deterministic leaderboard correction after Windows tests

The first user Windows test of the experiment013 suite returned
20 passed and one failed: the archive and promotion integration test
compared the leaderboard from a fresh two-worker run with that from
an identical verified-cache rerun. Both computed identical original
signed Max-Cut scores (10 in the tiny G11 synthetic case), but
worker chunks arrived in nondeterministic order. The former
aggregation iterated in arrival order, affecting the first returned
best-restart ID for tied scores and potentially floating-point
summation. Runtime was also an unstable score-tie ranking criterion.

The runner now sorts all trials by ORIGINAL restart ID before
computing mean, median, standard deviation, best-restart and time
aggregates. Leaderboard ranking uses (best cut descending, mean cut
descending, configuration ID ascending). Runtime is still measured
and reported but is not a tie-breaker. A synthetic regression test
explicitly reverses worker-completion order and requires byte-equivalent
leaderboard objects, including the deterministic restart tie.

The numerical optimization routines and saved binary partitions are
unchanged. Run these from the same current local branch after pulling:

    git pull --ff-only origin experiment013-eva-pipeline-grid
    python -m pytest tests/test_eva_pipeline_grid.py tests/test_eva_cutaware_kmeans.py -q

This correction has been committed to the experiment013 branch; the
full Windows rerun remains for the user to verify. Existing verified
archives can be reused without regenerating EVA factors or roundings.
