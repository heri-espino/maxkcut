# Experiment010 — Concurrent KL-style and Tabu refinement of archived EVA/K-means cuts

## Research question

Our saved G63 experiment008 found a **feasible cut of 26,682** after
1,000 independent antipodal K-MeansNM roundings of the previously
optimized EVA factor, each followed by positive-gain sparse vertex
flips. The earlier MST warm-start experiments did not improve it.

Can non-monotone local search starting from the SAME saved
K-MeansNM-Flip partitions escape those 1-flip local optima and improve
the original signed weighted Max-Cut?

We test **TWO independent algorithms simultaneously** (Windows-safe
separate processes), on paired identical source initial partitions:

1. **Generalized Kernighan–Lin (KL)-style gain-sequence search.**
   Lock each vertex once in a full pass, always select the currently
   highest-gain unlocked vertex (even when the gain is negative), and
   KEEP only the best strictly positive cumulative prefix. Repeat
   up to \`--kl-passes\` passes or until a pass has no positive prefix.
   This is **not** classic balanced 2-vertex KL swapping; it is an
   unbalanced, signed-weight single-vertex gain-sequence variant.
2. **Best-admissible Tabu search.**
   Select the highest-gain non-tabu move (which may worsen the current
   objective), with aspiration if a tabu move strictly exceeds the
   global best recorded score. A vertex is tabu for approximately
   \`--tabu-tenure\` moves, with independently seeded jitter.
   Return the **best-ever visited** partition, not the final
   possibly inferior search state. Stop at \`--tabu-steps\` or
   after \`--tabu-stagnation\` moves without improving the incumbent.

Both methods operate on the SAME original sparse signed G-set graph,
not on the EVA vector geometry or an MST. No second EVA optimization,
no new rounding random seeds and no repeated MST construction.

## Mathematical invariant and complexity

For binary spin labels \(s_i \in \{-1,+1\}\), the original Max-Cut is

\[
  C(s)=\frac12\sum_{(i,j)\in E} w_{ij}(1-s_is_j).
\]

Flipping \(s_i\) changes the cut by exactly

\[
  \Delta_i=s_i\sum_{j} w_{ij}s_j.
\]

After flipping i, update gains for the neighbors in O(deg i):
\(\Delta_j \leftarrow \Delta_j-2w_{ij}s_is_j\).

- KL pass: O(n²+m) operations in compiled Numba; each vertex moves
  at most once per pass. Max ~\`kl_passes*(n²+m)\` operations.
- Tabu: O(steps*n + sum of degrees of moved vertices) operations in
  compiled Numba; O(n+m) memory.
- Both independently re-score the resulting partition using the
  original graph, check that cumulative gains match observed
  differences, and assert \(C_\text{after}\ge C_\text{before}\).
- Passing the 1-flip local optimum is a plausible benefit of non-monotone
  moves; neither method guarantees a global optimum or a particular
  improvement.

The per-trial Numba compilation cost on Windows can occur once in
each worker; the initial benchmark should not mistake this startup
for steady-state solve time. The report distinguishes algorithm time
per trial. Two concurrent processes do not imply the total wall time
is half that of one process on every CPU configuration.

## Input provenance and fairness

Input is the **already saved experiment008** rounding-extension
partition archive under

    results/gset/experiment008-eva-rounding-flips/segments/
      G63-<factorSha16>/K-MeansNM-Flip_EVA/r00000-r01000.json

The loader checks the original graph SHA256, source factor SHA256,
source partition NPZ SHA256, original rounding seed and range, and
independently verifies the archived binary partitions against their
saved weighted graph. Ambiguous EVA factors and duplicate restart
indices are hard errors. If no input archive exists after switching
branches, **merge the previously committed branch containing
experiment008 artifacts** into the current branch; do not rerun EVA.

The same top \`--top-starts\` partitions ranked by their SAVED
actual Max-Cut scores are handed to KL and Tabu. Thus starts are
**paired**; selection is explicitly score-biased and not intended as
an unbiased estimate of the 1,000-restart score distribution.
Use \`--top-starts 0\` to use all archived inputs. A separate
\`--source-method FJ-Flip_EVA\` control uses GW/FJ-flip starts.

Each source restart index generates a distinct deterministic random
stream per method independent of scheduling, worker count, and
checkpoint chunk size. One source may recur with a complementary
partition; the archive preserves the original restart ID.

## Reproduction: PowerShell / G63

**Before switching branches**, save all experiment008 and experiment009
outputs on the CURRENT branch and wait for a successful push:

    git add .
    git commit -m "experiments008-009: preserve G63 optimized rounding and MST results"
    git push

Switch to the implementation branch:

    git fetch origin
    git switch --track origin/experiment010-kmeans-kl-tabu

If the branch already exists locally, use:

    git switch experiment010-kmeans-kl-tabu

If experiment008 output files were committed on a different branch
and are now absent, merge that exact pushed branch before executing.
Do not change branches while an experiment is still running.

Test on the user's Conda VM (Numba is required; install existing
\`eva-fast\` optional dependencies if necessary):

    $env:PYTHONPATH = Join-Path (Get-Location).Path "src"
    python -m pytest tests/test_eva_escape.py -q

The main G63 pilot with **200 strongest saved K-MeansNM-Flip starts
for EACH method**, running both concurrently on two processes,
at a substantial budget:

    python -m maxkcut.eva_escape_cli --graphs G63 --source-dir results/gset/experiment008-eva-rounding-flips --output-dir results/gset/experiment010-eva-kl-tabu --source-method K-MeansNM-Flip_EVA --methods KL Tabu --top-starts 200 --chunk-size 20 --workers 2 --seed 2026 --kl-passes 30 --tabu-steps 40000 --tabu-tenure 25 --tabu-stagnation 8000 --benchmark 26682

If performance is encouraging, cover **ALL 1,000** saved starts.
Previously finished top-200 chunks should verify and skip under the
same settings and chunking:

    python -m maxkcut.eva_escape_cli --graphs G63 --source-dir results/gset/experiment008-eva-rounding-flips --output-dir results/gset/experiment010-eva-kl-tabu --source-method K-MeansNM-Flip_EVA --methods KL Tabu --top-starts 1000 --chunk-size 20 --workers 2 --seed 2026 --kl-passes 30 --tabu-steps 40000 --tabu-tenure 25 --tabu-stagnation 8000 --benchmark 26682

To test more aggressive search while keeping comparable starting
partitions, raise the KL and Tabu budgets and change the seed:

    --kl-passes 60 --tabu-steps 100000 --tabu-stagnation 20000 --seed 2027

Each distinct configuration creates independently fingerprinted,
append-only output; scores from different budgets must NOT be
misrepresented as strictly matched computational cost. The reported
comparative unit is identical starting partition; algorithms may
have different running times.

After verifying output, commit/push everything from the VM:

    git add .
    git commit -m "experiment010: save G63 parallel KL and Tabu cut results"
    git push

The CLI also accepts \`--push --push-branch experiment010-kmeans-kl-tabu\`,
which stages/pushes the output directory **ONCE at the end**, after
all selected graphs. On an interrupted run, already written per-chunk
archives remain on disk but are not remotely preserved until committed
and pushed.

## Output and verification

- \`cases/G63-...-KL-<settingsHash>.json\` and corresponding Tabu case,
  one per source chunk, with immutable input fingerprint and every
  restart score/diagnostic.
- \`archives/<runId>/G63-k2/graph.npz\` and bit-packed refined binary
  partitions; each archive independently scores EVERY partition.
- \`trials/G63-...-<method>-<settingsHash>.jsonl\`, one record per
  identical input restart (initial, final, improvement, termination,
  seconds, algorithm-specific move/pass diagnostics).
- \`summary.csv\`, one row per source restart * method, including the
  score difference versus the recorded 26,682 benchmark.
- \`summaries/G63-K-MeansNM-Flip_EVA-<factorSha16>.json\`,
  count, best/mean/median/std, fraction improved, and aggregate runtime.
- \`errors/\` with per-chunk traceback diagnostics if tasks fail.

Independent source/rescoring checks catch mismatched graph/partition
artifacts. Output archives are verified immediately, and an identical
rerun verifies the existing result rather than silently overwriting.

## Interpretation

Success means the new heuristic discovers a better feasible G63
Max-Cut from some existing K-MeansNM-Flip warm start. A benefit from
Tabu/KL **cannot** by itself establish better EVA embeddings or a
new Max-Cut optimum, or superiority versus alternatives at unequal
time budgets. We should report at least:

- For each method, best, mean, median, sample standard deviation and
  fraction of selected starts improved.
- Paired per-source-restart gains, KL vs Tabu, and seconds per trial.
- For best discovered partitions, complete archive re-verification.
- As controls: the same search budgets starting from the saved
  FJ-Flip_EVA roundings and (if practical) random 1-flip local minima.

No completed G63 KL or Tabu results are claimed prior to running.
