# Experiment011 — Intensive KL elite search toward G63 score 27,047

## What has been observed

- Experiment008: best feasible G63 signed Max-Cut 26,682 from
  1,000 K-MeansNM-Flip EVA roundings.
- Experiment010: best KL-style locally optimized cut **26,886**
  (a +204 gain). Tabu best **26,688** (a +6 gain).
- Target **27,047**: 161 points above the best observed KL cut.
  This is a target, **NOT** a certified optimum or guarantee.

## New procedures

Reuse only the independently verified archived experiment010 **KL**
output partitions as initial conditions (the sources have graph SHA,
factor SHA, source label/checksum and original graph signed objective
verified). No EVA, MST, PCA or K-means recomputation.

**ILS-KL:** Repeated perturbations of the strongest KL incumbents,
varying flip count (8,16,32,64,128,256), with ~65% chosen from
small-absolute-gain boundary vertices and the rest uniformly random.
After each perturbation, apply KL. A walking state can probabilistically
accept worse refined candidates to explore further basins. Only the
highest-scoring partition seen is ever returned.

**EliteMix-KL:** Combine strong KL solutions after accounting for
binary-label complement equivalence. Recombine randomized fractions
of disagreeing vertices, then apply KL. If the partner is identical,
use a standard random/boundary kick.

**Tabu-KL:** Diversify the current state using independently seeded
Tabu moves with rotating tenure (0.5,1,1.5,2 times its base), then
refine by KL. This is an additional control, since the previous Tabu
performance was weaker.

**Fast KL:** A Numba indexed maximum heap replaces the previous
O(n²) gain maximum scan. Selecting all n vertices with neighbor
gain updates now takes O((n+m)log n) worst-case per KL pass and O(n+m)
memory. The same locked best-gain and positive-prefix semantics and
seeded tie order are retained. Synthetic signed-reference testing
validated exact correspondence of an equivalent heap implementation
to the former scan on 500 randomized cases. It is not a proof that
the user's Windows environment tests will pass.

All methods optimize the **original** undirected signed weighted graph.
The verified final objective cannot be less than its saved initial
KL cut. Any 27,047 cut would be a feasible solution, not proof of
optimality. Outputs contain every per-cycle candidate and best score.

## Branch and environment preparation

Save and push your experiment010 G63 results BEFORE switching branches,
and wait for the push to complete. Execute these in PowerShell:

    git add .
    git commit -m "experiment010: preserve G63 KL 26886 and Tabu results"
    git push

Then:

    git fetch origin
    git switch --track origin/experiment011-kl-elite-search

If already created locally, use:

    git switch experiment011-kl-elite-search

If the saved experiment010 results are on a different Git branch and
do not appear under the new branch, merge that pushed branch before
running. Never rerun EVA to recover a committed partition archive.

## Tests

    $env:PYTHONPATH = Join-Path (Get-Location).Path "src"
    python -m pytest tests/test_eva_elite_search.py tests/test_eva_escape.py -q

## Pilot: all three methods concurrently

    python -m maxkcut.eva_elite_cli --graphs G63 --methods ILS-KL EliteMix-KL Tabu-KL --top-elites 8 --replicates 2 --cycles 24 --kl-passes 8 --kick-sizes 8 16 32 64 128 --tabu-steps 3000 --tabu-stagnation 1000 --workers 3 --chunk-size 4 --target 27047 --source-dir results/gset/experiment010-eva-kl-tabu --output-dir results/gset/experiment011-elite-search

This pilots 16 episodes per method, in 3 independent processes,
using the same top eight distinct archived KL partitions for all
methods. All restarts and perturbations have deterministic per-episode
seeds, independent of process scheduling.

## Main large-budget experiment

    python -m maxkcut.eva_elite_cli --graphs G63 --methods ILS-KL EliteMix-KL --top-elites 24 --replicates 5 --cycles 80 --kl-passes 12 --kick-sizes 8 16 32 64 128 256 --temperature 12 --workers 4 --chunk-size 8 --target 27047 --source-dir results/gset/experiment010-eva-kl-tabu --output-dir results/gset/experiment011-elite-search

This allocates 120 independent episodes per method, **240 total**,
each up to 80 diversification cycles (up to **19,200 cycles**
of fast KL per two-method experiment). Increasing KL passes alone
is not the same as these randomized escape episodes.

## Additional Tabu-KL intensive control

    python -m maxkcut.eva_elite_cli --graphs G63 --methods Tabu-KL --top-elites 12 --replicates 3 --cycles 35 --kl-passes 12 --tabu-steps 5000 --tabu-tenure 15 --tabu-stagnation 1500 --workers 3 --chunk-size 6 --target 27047 --source-dir results/gset/experiment010-eva-kl-tabu --output-dir results/gset/experiment011-elite-search

When desired, scale the primary methods to
--top-elites 40 --replicates 8 --cycles 150 --kl-passes 20.
These are **substantially** larger searches and may be CPU-intensive,
despite cheap individual graph cuts. Do not promise a fixed duration.

## Exact reproducibility and archives

Input: experiment010 cases and verified original graph/partition
snapshots. Source KL score and factor checksum are required, with
source KL seed=2026 and pass setting=30 by default. Duplicated
source restart indices and ambiguous factor SHA versions are errors.
The top elites are distinct up to global partition complement.

Output: experiment011 results directory:
- cases/ : independent per-chunk JSON records (original score,
  improved score, full per-cycle trace, source restart, replicate,
  exact budgets, seed and archive identity).
- archives/ : original G-set weighted graph snapshot and bit-packed
  optimized binary partitions; independently rescored and verified.
- trials/ : JSONL with complete per-episode search trajectories.
- summary.csv : all archived episodes and distance from target 27,047.
- summaries/ : per-configuration aggregates including best, mean,
  median, standard deviation, improvement fraction, and CPU seconds.
- errors/ : per-chunk error diagnostics, plus graph-level errors.

A chunk of 8 episodes produces ONE graph archive to limit
duplicated weighted graph snapshots. Completed chunks are
independently verified and skipped on identical reruns; a partial
chunk must be recomputed. Changing budgets creates an additional,
immutable settings-fingerprinted run instead of replacing old results.
A single optional --push --push-branch experiment011-kl-elite-search
commits/pushes experiment outputs at the end. Interrupted runs must
be manually committed/pushed to preserve partial completed chunks.

Suggested results persistence after a run:

    git add .
    git commit -m "experiment011: archive extensive G63 elite searches"
    git push

## Interpretations and further work

If 27,047 is reached, report the exact score and verify its stored
partition under the original graph. It does **not** prove global
optimality. If progress stalls below target, inspect score-vs-cycle
trajectories and diversity across elite sources, then consider stronger
very-large-neighborhood search, exact optimization of small vertex
subsets via MILP, or alternate elite recombination rules. Respect
matched compute budget when comparing methods or reporting claims.

## Preliminary verification completed outside the user's VM

A standalone Python reference and separately compiled Numba version of
our indexed-heap KL pass were compared with the original O(n^2)
full-scan pass on randomized signed symmetric sparse graphs. The
Python heap comparison matched on 500 trials; the separately compiled
Numba heap comparison matched on 300 trials (n=4,7,10,18,40,80).
These checks validate the gain-selection/prefix algorithm but were not
imports from the connected GitHub repository and are not equivalent
to running the committed pytest suite in the user's Windows Conda
installation. The actual experiment011 full tests and G63 run remain
pending; no new G63 cut value is claimed.
